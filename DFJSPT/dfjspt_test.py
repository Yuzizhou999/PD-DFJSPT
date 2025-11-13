"""
Multi-Objective Evaluation Script for DFJSPT
功能等同于 PDMORL 仓库中的 MORL_utils.py 和 eval_benchmarks_MO_TD3_HER.py 的总和
这是一个全方位的多目标评估"考场"
"""

import json
import os
import time
import itertools
import pandas as pd
from ray.rllib import Policy


import gymnasium as gym
gymnasium = True
import ray
from ray.rllib.models import ModelCatalog
import numpy as np
import matplotlib.pyplot as plt
from DFJSPT import dfjspt_params
from DFJSPT.dfjspt_data.load_data_from_Ham import load_instance
from DFJSPT.dfjspt_env import DfjsptMaEnv
from DFJSPT.dfjspt_agent_model import JobActionMaskModel, MachineActionMaskModel, TransbotActionMaskModel

# ============== 从 MORL_utils.py 导入的辅助函数 (开始) ============== #

def generate_w_batch_eval(reward_size, step_size):
    """
    生成所有可能的偏好向量组合（用于评估）
    从 MORL_utils.py 的 generate_w_batch_test 改编
    
    Args:
        reward_size: 目标数量 (对于 DFJSPT 是 2: makespan 和 tardiness)
        step_size: 偏好向量的步长 (例如 0.1 会生成 [1.0,0.0], [0.9,0.1], ..., [0.0,1.0])
    
    Returns:
        w_batch_eval: numpy array of shape (n_prefs, reward_size)
    """
    mesh_array = []
    for i in range(reward_size):
        mesh_array.append(np.arange(0, 1 + step_size, step_size))
    
    # 生成所有组合
    w_batch_eval = np.array(list(itertools.product(*mesh_array)))
    
    # 只保留和为 1 的向量（单纯形约束）
    w_batch_eval = w_batch_eval[np.isclose(w_batch_eval.sum(axis=1), 1.0), :]
    
    # 去除重复
    w_batch_eval = np.unique(w_batch_eval, axis=0)
    
    return w_batch_eval


def compute_sparsity(obj_batch):
    """
    计算 Pareto 前沿的稀疏度（目标空间的均匀分布程度）
    从 MORL_utils.py 复制
    
    Args:
        obj_batch: numpy array of shape (n_solutions, n_objectives)
    
    Returns:
        sparsity: float, 稀疏度分数（越小越好，表示分布越均匀）
    """
    # 如果没有足够的解，返回 0
    if len(obj_batch) <= 1:
        return 0.0
    
    sparsity_sum = 0.0
    
    # 对每个目标维度
    for objective_idx in range(obj_batch.shape[-1]):
        objs_sort = np.sort(obj_batch[:, objective_idx])
        sp = 0.0
        # 计算相邻解之间的距离平方和
        for i in range(len(objs_sort) - 1):
            sp += np.power(objs_sort[i] - objs_sort[i + 1], 2)
        sparsity_sum += sp
    
    # 归一化
    sparsity = sparsity_sum / (len(obj_batch) - 1)
    
    return sparsity


def plot_objs(all_objectives, scenario_name="DFJSPT", ext='', save_path=None):
    """
    绘制 Pareto 前沿图
    从 MORL_utils.py 改编
    
    Args:
        all_objectives: numpy array of shape (n_solutions, 2), 注意是负值（越大越好）
        scenario_name: 场景名称
        ext: 文件名后缀
        save_path: 保存路径
    """
    # 因为我们的目标是负值（越大越好），绘图时取反以更直观
    objs_plot = -all_objectives  # 转换为正值（越小越好）
    
    plt.figure(figsize=(10, 8))
    plt.plot(objs_plot[:, 0], objs_plot[:, 1], 'ro', markersize=8, alpha=0.6, label='Pareto Solutions')
    plt.xlabel("Makespan", fontsize=14)
    plt.ylabel("Total Tardiness", fontsize=14)
    plt.title(f'{scenario_name} - Pareto Front', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    if save_path is None:
        save_path = f'Figures/{scenario_name}-ParetoFront_{ext}.png'
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Pareto front plot saved to: {save_path}")
    plt.close()

# ============== 从 MORL_utils.py 导入的辅助函数 (结束) ============== #


def eval_agent_multi_objective(env, job_policy, machine_policy, transbot_policy, w_batch, num_eval_episodes=10, test_instances=10):
    """
    多目标评估函数
    功能等同于 eval_benchmarks_MO_TD3_HER.py 中的 eval_agent
    
    Args:
        env: DFJSPT 环境
        job_policy, machine_policy, transbot_policy: 三个决策层的策略
        w_batch: 偏好向量集合 (n_prefs, 2)
        num_eval_episodes: 每个偏好运行的次数（用于平均）
        test_instances: 测试实例数量
    
    Returns:
        all_objectives: (n_prefs, 2) 每个偏好的平均目标值
        hypervolume: float, 超体积分数
        sparsity: float, 稀疏度分数
    """
    print(f"\n{'='*80}")
    print(f"开始多目标评估")
    print(f"偏好向量数量: {len(w_batch)}")
    print(f"每个偏好运行次数: {num_eval_episodes}")
    print(f"测试实例数量: {test_instances}")
    print(f"{'='*80}\n")
    
    # 存储所有偏好的目标值
    all_objectives = []
    
    # 遍历所有偏好向量（外循环）
    for pref_idx, w in enumerate(w_batch):
        print(f"\n[偏好 {pref_idx+1}/{len(w_batch)}] 正在评估偏好: {w}")
        
        # 为当前偏好收集多次运行的目标值
        episode_objectives = []
        
        # 内循环：为每个偏好运行 N 次（多个测试实例）
        for instance_id in range(test_instances):
            # 关键：调用 reset 并传入"考题"（指定偏好向量）
            observation, info = env.reset(options={
                "eval_preference": w,
                "instance_id": instance_id,
            })
            
            done = False
            stage = next(iter(info["agent0"].values()), None)
            step_count = 0
            
            # 运行一个完整的 Episode
            while not done:
                if stage == 0:
                    # Job agent 决策
                    job_action = {
                        "agent0": job_policy.compute_single_action(obs=observation["agent0"], explore=False)[0]
                    }
                    observation, reward, terminated, truncated, info = env.step(job_action)
                    stage = next(iter(info["agent1"].values()), None)
                
                elif stage == 1:
                    # Machine agent 决策
                    machine_action = {
                        "agent1": machine_policy.compute_single_action(obs=observation["agent1"], explore=False)[0]
                    }
                    observation, reward, terminated, truncated, info = env.step(machine_action)
                    stage = next(iter(info["agent2"].values()), None)
                
                else:
                    # Transbot agent 决策
                    transbot_action = {
                        "agent2": transbot_policy.compute_single_action(obs=observation["agent2"], explore=False)[0]
                    }
                    observation, reward, terminated, truncated, info = env.step(transbot_action)
                    stage = next(iter(info["agent0"].values()), None)
                    done = terminated["__all__"]
                    step_count += 1
            
            # 记录最终成绩（从 info 中获取）
            if "objectives" in info["agent0"]:
                final_objs = info["agent0"]["objectives"]
            else:
                # 如果 info 中没有，从环境中直接获取
                final_objs = np.array([
                    -env.final_makespan,
                    -env.curr_tardiness
                ], dtype=np.float32)
            
            episode_objectives.append(final_objs)
            
            if (instance_id + 1) % 5 == 0:
                print(f"  Instance {instance_id+1}/{test_instances} 完成")
        
        # 计算当前偏好的平均成绩
        avg_objectives = np.mean(episode_objectives, axis=0)
        all_objectives.append(avg_objectives)
        
        print(f"  平均目标值: Makespan={-avg_objectives[0]:.2f}, Tardiness={-avg_objectives[1]:.2f}")
    
    # 考试结束
    all_objectives = np.array(all_objectives)
    
    print(f"\n{'='*80}")
    print(f"评估完成！")
    print(f"所有偏好的平均最终目标值 shape: {all_objectives.shape}")
    print(f"{'='*80}\n")
    
    # 计算"综合分" (Hypervolume)
    # 关键修正：我们的目标是负值（越大越好），需要转换为最小化问题
    # pymoo 的 HV 计算假设目标是最小化的，所以我们需要取反
    
    try:
        from pymoo.indicators.hv import HV
        
        # 转换为最小化问题：取反（负号变正号）
        # 原始：[-893.74, -2562.16] → 转换后：[893.74, 2562.16]
        all_objectives_min = -all_objectives  # 取反，转为正值（最小化）
        
        print(f"目标值范围（最小化形式）:")
        print(f"  Makespan: [{np.min(all_objectives_min[:, 0]):.2f}, {np.max(all_objectives_min[:, 0]):.2f}]")
        print(f"  Tardiness: [{np.min(all_objectives_min[:, 1]):.2f}, {np.max(all_objectives_min[:, 1]):.2f}]")
        
        # 参考点：设置为比所有解都差的点（在最小化问题中，就是更大的值）
        ref_point = np.array([
            np.max(all_objectives_min[:, 0]) * 1.1,  # Makespan 参考点（比最大值大 10%）
            np.max(all_objectives_min[:, 1]) * 1.1   # Tardiness 参考点（比最大值大 10%）
        ])
        
        print(f"参考点 (Reference Point): [{ref_point[0]:.2f}, {ref_point[1]:.2f}]")
        
        ind = HV(ref_point=ref_point)
        hv_score = ind(all_objectives_min)
        
        print(f"超体积 (Hypervolume): {hv_score:.4f}")
    except ImportError:
        print("警告: pymoo 未安装，无法计算 Hypervolume")
        hv_score = 0.0
    except Exception as e:
        print(f"警告: Hypervolume 计算失败: {e}")
        print(f"详细错误信息: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        hv_score = 0.0
    
    # 计算"平滑度" (Sparsity)
    # 同样使用转换后的目标值（正值）来计算，使数值更有意义
    all_objectives_min = -all_objectives  # 转为正值
    sparsity_score = compute_sparsity(all_objectives_min)
    print(f"稀疏度 (Sparsity): {sparsity_score:.4f}")
    
    return all_objectives, hv_score, sparsity_score


def main():
    """主评估流程"""
    
    # ========== 初始化 Ray ========== #
    ray.init(local_mode=False)
    
    time0 = time.time()
    
    # ========== 注册自定义模型 ========== #
    ModelCatalog.register_custom_model("job_agent_model", JobActionMaskModel)
    ModelCatalog.register_custom_model("machine_agent_model", MachineActionMaskModel)
    ModelCatalog.register_custom_model("transbot_agent_model", TransbotActionMaskModel)
    
    # ========== 加载 Agent ========== #
    # TODO: 修改为你想要评估的 checkpoint 路径
    checkpoint_path = os.path.dirname(__file__) + "/training_results/J" + str(
        dfjspt_params.n_jobs) + "_M" + str(dfjspt_params.n_machines) + "_T" + str(
        dfjspt_params.n_transbots) + '/MyTrainable_DfjsptMaEnv_PDMORL_4a8c7_00000_0_2025-11-10_10-02-20/checkpoint_000258'
    
    print(f"正在加载 checkpoint: {checkpoint_path}")
    
    job_policy_checkpoint_path = checkpoint_path + '/policies/policy_agent0'
    job_policy = Policy.from_checkpoint(job_policy_checkpoint_path)
    
    machine_policy_checkpoint_path = checkpoint_path + '/policies/policy_agent1'
    machine_policy = Policy.from_checkpoint(machine_policy_checkpoint_path)
    
    transbot_policy_checkpoint_path = checkpoint_path + '/policies/policy_agent2'
    transbot_policy = Policy.from_checkpoint(transbot_policy_checkpoint_path)
    
    time1 = time.time()
    print(f"✓ 策略加载完成，耗时 {time1-time0:.2f} 秒\n")
    
    # ========== 生成"考卷" (偏好向量集合) ========== #
    REWARD_SIZE = 2  # Makespan 和 Tardiness
    W_STEP_SIZE = 0.05  # 偏好向量步长（可调整精细度）
    
    print(f"正在生成偏好向量集合（步长={W_STEP_SIZE}）...")
    w_batch_eval = generate_w_batch_eval(REWARD_SIZE, W_STEP_SIZE)
    print(f"✓ 生成了 {len(w_batch_eval)} 个偏好向量")
    print(f"偏好向量示例: {w_batch_eval[:5]}\n")
    
    # ========== 创建环境 ========== #
    env = DfjsptMaEnv({
        "train_or_eval_or_test": "test",
    })
    
    # ========== 开始多目标评估 ========== #
    NUM_EVAL_EPISODES = 10  # 每个偏好运行 N 个测试实例
    TEST_INSTANCES = dfjspt_params.n_instances_for_testing
    
    time2 = time.time()
    
    all_objectives, hv_score, sparsity_score = eval_agent_multi_objective(
        env=env,
        job_policy=job_policy,
        machine_policy=machine_policy,
        transbot_policy=transbot_policy,
        w_batch=w_batch_eval,
        num_eval_episodes=NUM_EVAL_EPISODES,
        test_instances=TEST_INSTANCES
    )
    
    time3 = time.time()
    print(f"\n评估总耗时: {time3-time2:.2f} 秒")
    
    # ========== 保存结果 ========== #
    results_dir = os.path.join(os.path.dirname(checkpoint_path), "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存目标值到 CSV
    results_df = pd.DataFrame(
        all_objectives,
        columns=["Makespan (negative)", "Total Tardiness (negative)"]
    )
    results_df["Preference_Weight_Makespan"] = w_batch_eval[:, 0]
    results_df["Preference_Weight_Tardiness"] = w_batch_eval[:, 1]
    
    csv_path = os.path.join(results_dir, "pareto_front_objectives.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ 目标值已保存到: {csv_path}")
    
    # 保存评估指标
    metrics = {
        "hypervolume": float(hv_score),
        "sparsity": float(sparsity_score),
        "num_preferences": len(w_batch_eval),
        "num_eval_episodes": NUM_EVAL_EPISODES,
        "test_instances": TEST_INSTANCES,
        "checkpoint_path": checkpoint_path,
    }
    
    metrics_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ 评估指标已保存到: {metrics_path}")
    
    # ========== 绘制 Pareto 前沿 ========== #
    plot_path = os.path.join(results_dir, "pareto_front.png")
    plot_objs(all_objectives, scenario_name="DFJSPT", ext="eval", save_path=plot_path)
    
    # ========== 打印总结 ========== #
    print(f"\n{'='*80}")
    print(f"评估总结")
    print(f"{'='*80}")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"偏好向量数量: {len(w_batch_eval)}")
    print(f"测试实例数量: {TEST_INSTANCES}")
    print(f"超体积 (Hypervolume): {hv_score:.4f}")
    print(f"稀疏度 (Sparsity): {sparsity_score:.4f}")
    print(f"平均 Makespan: {-np.mean(all_objectives[:, 0]):.2f}")
    print(f"平均 Tardiness: {-np.mean(all_objectives[:, 1]):.2f}")
    print(f"{'='*80}\n")
    
    # ========== 清理 ========== #
    ray.shutdown()


if __name__ == "__main__":
    main()

