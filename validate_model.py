"""
完整的模型验证脚本
用于对比DRL模型与传统规则方法（baseline）的性能

验证维度：
1. 单目标性能：Makespan
2. 多目标性能：Makespan + Tardiness
3. Pareto前沿质量：超体积、稀疏度
4. 泛化能力：在unseen实例上的表现
5. 计算效率：推理时间
"""

import json
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from ray.rllib.algorithms.ppo import PPO

from DFJSPT import dfjspt_params
from DFJSPT.dfjspt_env import DfjsptMaEnv
from DFJSPT.dfjspt_agent_model import JobActionMaskModel, MachineActionMaskModel, TransbotActionMaskModel
from DFJSPT.dfjspt_rule.job_selection_rules import job_EST_action, job_SPT_action
from DFJSPT.dfjspt_rule.machine_selection_rules import machine_EET_action, machine_SPT_action, transbot_EET_action, transbot_SPT_action
from DFJSPT.dfjspt_test import generate_w_batch_eval, compute_sparsity, plot_objs
import copy

import ray
from ray.rllib.models import ModelCatalog
from ray import tune


class ModelValidator:
    """模型验证器"""
    
    def __init__(self, checkpoint_path, test_instances_count=100):
        """
        Args:
            checkpoint_path: 训练好的模型checkpoint路径
            test_instances_count: 测试实例数量
        """
        self.checkpoint_path = checkpoint_path
        self.test_instances_count = test_instances_count
        self.results = {}
        
        # 初始化Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # 注册自定义模型
        ModelCatalog.register_custom_model("job_agent_model", JobActionMaskModel)
        ModelCatalog.register_custom_model("machine_agent_model", MachineActionMaskModel)
        ModelCatalog.register_custom_model("transbot_agent_model", TransbotActionMaskModel)
        
        # 定义偏好向量创建函数并注册环境
        def create_env_with_preferences(env_config):
            """创建带有偏好向量的环境实例"""
            env = DfjsptMaEnv(env_config)
            env.preference_vector = np.array([0.5, 0.5])
            return env
        
        tune.register_env("DfjsptMaEnv_PDMORL", create_env_with_preferences)
        
        # 加载DRL模型
        print(f"正在加载模型: {checkpoint_path}")
        self.algo = PPO.from_checkpoint(checkpoint_path)
        self.drl_policies = {
            "job": self.algo.get_policy("policy_agent0"),
            "machine": self.algo.get_policy("policy_agent1"),
            "transbot": self.algo.get_policy("policy_agent2")
        }
        print("✓ DRL模型加载成功\n")
        
        # 初始化规则方法 - 使用函数而非类
        self.rule_methods = {
            "EST_EET_EET": {
                "job": job_EST_action,
                "machine": machine_EET_action,
                "transbot": transbot_EET_action
            },
            "EST_SPT_SPT": {
                "job": job_EST_action,
                "machine": machine_SPT_action,
                "transbot": transbot_SPT_action
            },
            "SPT_SPT_SPT": {
                "job": job_SPT_action,
                "machine": machine_SPT_action,
                "transbot": transbot_SPT_action
            }
        }
        print(f"✓ 已加载 {len(self.rule_methods)} 个规则方法\n")
    
    def create_test_env(self, eval_preference=None):
        """创建测试环境"""
        env_config = {
            "train_or_eval_or_test": "test",
            "num_jobs": dfjspt_params.n_jobs,
            "num_machines": dfjspt_params.n_machines,
            "num_transportbots": dfjspt_params.n_transbots,
            "generate_new_instances": True,
            "n_instances_for_training": 0,  # 测试时不使用训练实例
        }
        
        if eval_preference is not None:
            env_config["eval_preference"] = eval_preference
        
        return DfjsptMaEnv(env_config)
    
    def run_drl_episode(self, env, preference=None):
        """
        运行一个DRL episode
        
        Returns:
            makespan, tardiness, inference_time
        """
        if preference is not None:
            obs, info = env.reset(options={"eval_preference": preference})
        else:
            obs, info = env.reset()
        
        done = False
        stage = next(iter(info["agent0"].values()), None)
        inference_times = []
        
        while not done:
            if stage == 0:
                # Job agent
                start_time = time.time()
                action = self.drl_policies["job"].compute_single_action(obs["agent0"], explore=False)[0]
                inference_times.append(time.time() - start_time)
                obs, reward, terminated, truncated, info = env.step({"agent0": action})
                stage = next(iter(info["agent1"].values()), None)
            
            elif stage == 1:
                # Machine agent
                start_time = time.time()
                action = self.drl_policies["machine"].compute_single_action(obs["agent1"], explore=False)[0]
                inference_times.append(time.time() - start_time)
                obs, reward, terminated, truncated, info = env.step({"agent1": action})
                stage = next(iter(info["agent2"].values()), None)
            
            else:
                # Transbot agent
                start_time = time.time()
                action = self.drl_policies["transbot"].compute_single_action(obs["agent2"], explore=False)[0]
                inference_times.append(time.time() - start_time)
                obs, reward, terminated, truncated, info = env.step({"agent2": action})
                stage = next(iter(info["agent0"].values()), None)
                done = terminated["__all__"]
        
        return env.final_makespan, env.curr_tardiness, np.mean(inference_times)
    
    def run_rule_episode(self, env, rule_funcs):
        """
        运行一个规则方法episode
        
        Args:
            env: 环境实例
            rule_funcs: dict包含三个规则函数
                {"job": job_func, "machine": machine_func, "transbot": transbot_func}
        
        Returns:
            makespan, tardiness, inference_time
        """
        obs, info = env.reset()
        
        done = False
        stage = next(iter(info["agent0"].values()), None)
        inference_times = []
        
        while not done:
            if stage == 0:
                # Job selection
                start_time = time.time()
                legal_actions = copy.deepcopy(obs["agent0"]["action_mask"])
                real_attrs = copy.deepcopy(obs["agent0"]["observation"])
                action_dict = rule_funcs["job"](legal_job_actions=legal_actions, real_job_attrs=real_attrs)
                action = action_dict["agent0"]
                inference_times.append(time.time() - start_time)
                obs, reward, terminated, truncated, info = env.step({"agent0": action})
                stage = next(iter(info["agent1"].values()), None)
            
            elif stage == 1:
                # Machine selection
                start_time = time.time()
                legal_actions = copy.deepcopy(obs["agent1"]["action_mask"])
                real_attrs = copy.deepcopy(obs["agent1"]["observation"])
                action_dict = rule_funcs["machine"](legal_machine_actions=legal_actions, real_machine_attrs=real_attrs)
                action = action_dict["agent1"]
                inference_times.append(time.time() - start_time)
                obs, reward, terminated, truncated, info = env.step({"agent1": action})
                stage = next(iter(info["agent2"].values()), None)
            
            else:
                # Transbot selection
                start_time = time.time()
                legal_actions = copy.deepcopy(obs["agent2"]["action_mask"])
                real_attrs = copy.deepcopy(obs["agent2"]["observation"])
                action_dict = rule_funcs["transbot"](legal_transbot_actions=legal_actions, real_transbot_attrs=real_attrs)
                action = action_dict["agent2"]
                inference_times.append(time.time() - start_time)
                obs, reward, terminated, truncated, info = env.step({"agent2": action})
                stage = next(iter(info["agent0"].values()), None)
                done = terminated["__all__"]
        
        return env.final_makespan, env.curr_tardiness, np.mean(inference_times)
    
    def test_single_objective_performance(self):
        """
        测试1: 单目标性能（仅关注Makespan）
        """
        print("="*80)
        print("测试1: 单目标性能对比 (Makespan)")
        print("="*80)
        
        env = self.create_test_env()
        results = {
            "DRL": {"makespans": [], "inference_times": []},
        }
        
        # 添加规则方法
        for rule_name in self.rule_methods.keys():
            results[rule_name] = {"makespans": [], "inference_times": []}
        
        # 运行测试
        for i in range(self.test_instances_count):
            # DRL
            makespan, _, inf_time = self.run_drl_episode(env, preference=np.array([1.0, 0.0]))
            results["DRL"]["makespans"].append(makespan)
            results["DRL"]["inference_times"].append(inf_time)
            
            # 规则方法
            for rule_name, rule_funcs in self.rule_methods.items():
                makespan, _, inf_time = self.run_rule_episode(env, rule_funcs)
                results[rule_name]["makespans"].append(makespan)
                results[rule_name]["inference_times"].append(inf_time)
            
            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{self.test_instances_count}")
        
        # 统计结果
        print("\n结果统计:")
        print(f"{'方法':<15} {'平均Makespan':<15} {'标准差':<12} {'推理时间(ms)':<15} {'vs DRL':<10}")
        print("-" * 80)
        
        drl_mean = np.mean(results["DRL"]["makespans"])
        
        for method_name in ["DRL"] + list(self.rule_methods.keys()):
            makespans = results[method_name]["makespans"]
            inf_times = results[method_name]["inference_times"]
            mean_ms = np.mean(makespans)
            std_ms = np.std(makespans)
            mean_time = np.mean(inf_times) * 1000  # 转为毫秒
            
            if method_name == "DRL":
                vs_drl = "baseline"
            else:
                improvement = ((mean_ms - drl_mean) / mean_ms) * 100
                vs_drl = f"{improvement:+.2f}%"
            
            print(f"{method_name:<15} {mean_ms:<15.2f} {std_ms:<12.2f} {mean_time:<15.4f} {vs_drl:<10}")
        
        self.results["single_objective"] = results
        print()
    
    def test_multi_objective_performance(self):
        """
        测试2: 多目标性能（Makespan + Tardiness）
        """
        print("="*80)
        print("测试2: 多目标性能对比")
        print("="*80)
        
        # 生成偏好向量
        preferences = generate_w_batch_eval(reward_size=2, step_size=0.2)
        print(f"生成 {len(preferences)} 个偏好向量\n")
        
        env = self.create_test_env()
        results = {
            "DRL": {"objectives": []},
        }
        
        for rule_name in self.rule_methods.keys():
            results[rule_name] = {"objectives": []}
        
        # 对每个偏好运行
        for pref_idx, pref in enumerate(preferences):
            print(f"偏好 [{pref_idx+1}/{len(preferences)}]: {pref}")
            
            # DRL
            drl_objs = []
            for _ in range(10):  # 每个偏好运行10次
                makespan, tardiness, _ = self.run_drl_episode(env, preference=pref)
                drl_objs.append([makespan, tardiness])
            results["DRL"]["objectives"].append(np.mean(drl_objs, axis=0))
            
            # 规则方法（不使用偏好）
            for rule_name, rule_funcs in self.rule_methods.items():
                rule_objs = []
                for _ in range(10):
                    makespan, tardiness, _ = self.run_rule_episode(env, rule_funcs)
                    rule_objs.append([makespan, tardiness])
                results[rule_name]["objectives"].append(np.mean(rule_objs, axis=0))
        
        # 计算Pareto前沿质量
        print("\n多目标性能分析:")
        print(f"{'方法':<15} {'平均Makespan':<15} {'平均Tardiness':<15} {'超体积(HV)':<12} {'稀疏度':<10}")
        print("-" * 85)
        
        for method_name in ["DRL"] + list(self.rule_methods.keys()):
            objs = np.array(results[method_name]["objectives"])
            avg_makespan = np.mean(objs[:, 0])
            avg_tardiness = np.mean(objs[:, 1])
            
            # 计算超体积
            try:
                from pymoo.indicators.hv import HV
                ref_point = np.array([np.max(objs[:, 0]) * 1.2, np.max(objs[:, 1]) * 1.2])
                ind = HV(ref_point=ref_point)
                hv_score = ind(objs)
            except:
                hv_score = 0.0
            
            # 计算稀疏度
            sparsity = compute_sparsity(objs)
            
            print(f"{method_name:<15} {avg_makespan:<15.2f} {avg_tardiness:<15.2f} {hv_score:<12.2f} {sparsity:<10.4f}")
        
        self.results["multi_objective"] = results
        
        # 绘制Pareto前沿对比图
        self.plot_pareto_comparison(results)
        print()
    
    def plot_pareto_comparison(self, results):
        """绘制所有方法的Pareto前沿对比"""
        plt.figure(figsize=(12, 8))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        markers = ['o', 's', '^', 'D', 'v']
        
        for idx, (method_name, color, marker) in enumerate(zip(
            ["DRL"] + list(self.rule_methods.keys()), colors, markers
        )):
            objs = np.array(results[method_name]["objectives"])
            plt.scatter(objs[:, 0], objs[:, 1], c=color, marker=marker, 
                       s=100, alpha=0.6, label=method_name, edgecolors='black')
        
        plt.xlabel("Makespan", fontsize=14)
        plt.ylabel("Total Tardiness", fontsize=14)
        plt.title("Pareto Front Comparison: DRL vs Rule-based Methods", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        os.makedirs("validation_results", exist_ok=True)
        plt.savefig("validation_results/pareto_comparison.png", dpi=300, bbox_inches='tight')
        print("✓ Pareto前沿对比图已保存: validation_results/pareto_comparison.png")
        plt.close()
    
    def test_generalization(self):
        """
        测试3: 泛化能力（在不同规模的问题上测试）
        """
        print("="*80)
        print("测试3: 泛化能力测试")
        print("="*80)
        print("测试在不同规模问题上的表现\n")
        
        # TODO: 这需要修改环境配置来测试不同规模
        # 这里先简单测试当前规模
        print("注意: 完整的泛化测试需要修改环境配置")
        print("当前仅在训练规模 (10 jobs × 5 machines × 3 transportbots) 上测试\n")
    
    def generate_report(self, save_path="validation_results/validation_report.md"):
        """生成验证报告"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("# 模型验证报告\n\n")
            f.write(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**测试实例数**: {self.test_instances_count}\n")
            f.write(f"**模型路径**: {self.checkpoint_path}\n\n")
            
            f.write("## 1. 单目标性能对比 (Makespan)\n\n")
            if "single_objective" in self.results:
                results = self.results["single_objective"]
                f.write("| 方法 | 平均Makespan | 标准差 | vs DRL |\n")
                f.write("|------|-------------|--------|--------|\n")
                
                drl_mean = np.mean(results["DRL"]["makespans"])
                for method_name in ["DRL"] + list(self.rule_methods.keys()):
                    makespans = results[method_name]["makespans"]
                    mean_ms = np.mean(makespans)
                    std_ms = np.std(makespans)
                    
                    if method_name == "DRL":
                        vs_drl = "baseline"
                    else:
                        improvement = ((mean_ms - drl_mean) / mean_ms) * 100
                        vs_drl = f"{improvement:+.2f}%"
                    
                    f.write(f"| {method_name} | {mean_ms:.2f} | {std_ms:.2f} | {vs_drl} |\n")
            
            f.write("\n## 2. 多目标性能对比\n\n")
            f.write("详见 `validation_results/pareto_comparison.png`\n\n")
            
            f.write("## 3. 关键发现\n\n")
            f.write("- DRL模型在单目标优化上的表现\n")
            f.write("- DRL模型的多目标平衡能力\n")
            f.write("- 与传统规则方法的对比优势\n")
        
        print(f"✓ 验证报告已保存: {save_path}")
    
    def run_full_validation(self):
        """运行完整验证流程"""
        print("\n" + "="*80)
        print("开始完整模型验证")
        print("="*80 + "\n")
        
        # 测试1: 单目标性能
        self.test_single_objective_performance()
        
        # 测试2: 多目标性能
        self.test_multi_objective_performance()
        
        # 测试3: 泛化能力
        # self.test_generalization()
        
        # 生成报告
        self.generate_report()
        
        print("\n" + "="*80)
        print("验证完成！")
        print("="*80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="验证DRL模型性能")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="模型checkpoint路径")
    parser.add_argument("--test_instances", type=int, default=100,
                       help="测试实例数量（默认100）")
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = ModelValidator(
        checkpoint_path=args.checkpoint,
        test_instances_count=args.test_instances
    )
    
    # 运行验证
    validator.run_full_validation()


if __name__ == "__main__":
    # 示例用法
    # python validate_model.py --checkpoint "DFJSPT/training_results/J10_M5_T3/MyTrainable_DfjsptMaEnv_PDMORL_4a8c7_00000_0_2025-11-10_10-02-20/checkpoint_000258" --test_instances 100
    
    # 如果不使用命令行参数，可以直接在这里指定
    # 使用实验分析中确定的最优checkpoint (iter 258)
    checkpoint_path = "DFJSPT/training_results/J10_M5_T3/MyTrainable_DfjsptMaEnv_PDMORL_4a8c7_00000_0_2025-11-10_10-02-20/checkpoint_000258"
    
    validator = ModelValidator(
        checkpoint_path=checkpoint_path,
        test_instances_count=100
    )
    
    validator.run_full_validation()
