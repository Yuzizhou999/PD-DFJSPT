import json
import os
import warnings
import time
import itertools
import numpy as np
import ray
from ray import air, tune, train
from ray.rllib.algorithms import PPOConfig, PPO
from ray.rllib.models import ModelCatalog
# from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
# from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
# from ray.rllib.algorithms.algorithm import Algorithm
from typing import Dict
# from gymnasium import spaces
# import torch
# import numpy as np

# import sys
# import os
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)

from DFJSPT import dfjspt_params
from DFJSPT.dfjspt_env import DfjsptMaEnv
from DFJSPT.dfjspt_agent_model import JobActionMaskModel, MachineActionMaskModel, TransbotActionMaskModel

# 原来就有的 base_log_dir
base_log_dir = os.path.join(
    os.path.dirname(__file__),
    f"training_results/J{dfjspt_params.n_jobs}_M{dfjspt_params.n_machines}_T{dfjspt_params.n_transbots}"
)
os.makedirs(base_log_dir, exist_ok=True)

# 用环境变量给名字更可控；否则用时间戳保证唯一
experiment_name = os.environ.get("DFJSPT_EXPERIMENT_NAME",
                                 time.strftime("exp_%Y%m%d_%H%M%S"))

def postprocess_her_trajectory(
        policy: Policy,
        batch: SampleBatch,
        *,
        reward_size: int | None = None,
):
    """Augment a trajectory with hindsight preference relabelling.

    Args:
        policy: Policy owning the trajectory (used for config lookup).
        batch: The postprocessed sample batch.
        reward_size: Optional override for the preference dimensionality.

    Returns:
        SampleBatch containing the concatenated original and hindsight data.
    """
    if batch.count == 0 or not batch[SampleBatch.DONES][-1]:
        return batch

    terminal_info = batch[SampleBatch.INFOS][-1]
    if not isinstance(terminal_info, dict):
        return batch

    objectives = terminal_info.get("objectives")
    if objectives is None:
        return batch

    objectives = np.asarray(objectives, dtype=np.float32).reshape(-1)

    config = getattr(policy, "config", {}) if policy is not None else {}
    if reward_size is None:
        reward_size = (config.get("env_config", {}) or {}).get("reward_size")
        if reward_size is None:
            reward_size = (config.get("model", {}).get("custom_model_config", {})
                           .get("reward_size"))
    if reward_size is not None:
        reward_size = int(reward_size)

    obs_preferences = None
    if isinstance(batch[SampleBatch.OBS], dict):
        obs_preferences = batch[SampleBatch.OBS].get("preference")

    if reward_size is None and isinstance(obs_preferences, np.ndarray):
        reward_size = obs_preferences.shape[-1]

    if reward_size is None:
        reward_size = objectives.shape[0]

    if objectives.shape[0] != reward_size:
        reward_size = min(reward_size, objectives.shape[0])
        objectives = objectives[:reward_size]

    abs_objectives = np.abs(objectives)
    denom = abs_objectives.sum()
    if denom <= np.finfo(np.float32).eps:
        w_hindsight = np.full(reward_size, 1.0 / reward_size, dtype=np.float32)
    else:
        w_hindsight = abs_objectives / denom

    her_batch = batch.copy()

    if isinstance(her_batch[SampleBatch.OBS], dict):
        obs_dict = dict(her_batch[SampleBatch.OBS])
        preference_values = obs_dict.get("preference")
        if isinstance(preference_values, np.ndarray):
            pref_batch = preference_values.shape[0]
            obs_dict["preference"] = np.repeat(
                w_hindsight.reshape(1, -1), pref_batch, axis=0
            ).astype(preference_values.dtype)
            her_batch[SampleBatch.OBS] = obs_dict

    her_rewards = her_batch[SampleBatch.REWARDS]
    her_rewards[-1] = float(np.dot(w_hindsight, objectives))
    her_batch[SampleBatch.REWARDS] = her_rewards

    her_infos = list(her_batch[SampleBatch.INFOS])
    if her_infos:
        last_info = her_infos[-1]
        if isinstance(last_info, dict):
            updated_info = dict(last_info)
            updated_info["current_w"] = w_hindsight.astype(np.float32)
            her_infos[-1] = updated_info
    her_batch[SampleBatch.INFOS] = np.array(her_infos, dtype=object)

    return concat_samples([batch, her_batch])

class DfjsptAgentModel:
    """Namespace object for DFJSPT custom model registrations."""

    JOB = "job_agent_model"
    MACHINE = "machine_agent_model"
    TRANSBOT = "transbot_agent_model"
    _registered = False

    @classmethod
    def register_all(cls):
        if cls._registered:
            return

        ModelCatalog.register_custom_model(cls.JOB, JobActionMaskModel)
        ModelCatalog.register_custom_model(cls.MACHINE, MachineActionMaskModel)
        ModelCatalog.register_custom_model(cls.TRANSBOT, TransbotActionMaskModel)
        cls._registered = True


DfjsptAgentModel.register_all()

def generate_w_batch(reward_size, step_size):
    """
    生成偏好向量批次 (从 MORL_utils.py 的 generate_w_batch_test 函数改编)
    
    参数:
        reward_size: 目标数量（维度）
        step_size: 采样精细度（例如 0.1 表示每个维度以 0.1 为步长）
    
    返回:
        w_batch: 偏好向量数组，每一行是一个偏好向量，所有元素和为 1
    """
    mesh_array = []
    for i in range(reward_size):
        mesh_array.append(np.arange(0, 1 + step_size, step_size))
    
    w_batch = np.array(list(itertools.product(*mesh_array)))
    # 只保留和为 1 的偏好向量
    w_batch = w_batch[w_batch.sum(axis=1) == 1, :]
    # 去除重复
    w_batch = np.unique(w_batch, axis=0)
    
    return w_batch


def create_env_with_preferences(env_config):
    """
    自定义环境创建器，为每个 Worker 分配不同的偏好集合
    
    参数:
        env_config: 环境配置字典，由 RLlib 传递
    
    返回:
        DfjsptMaEnv: 配置好偏好集合的环境实例
    """
    # 获取 Worker 信息
    worker_index = env_config.worker_index
    num_workers = env_config.num_workers
    
    # 定义目标数量和偏好采样精细度
    REWARD_SIZE = 2
    W_STEP_SIZE = 0.05
    
    # 生成完整的偏好集合
    full_w_batch = generate_w_batch(REWARD_SIZE, W_STEP_SIZE)
    
    # 根据 Worker 数量分割偏好集
    if num_workers > 0:
        w_splits = np.array_split(full_w_batch, num_workers)
        
        # Worker 0 是主进程（用于评估），分配完整的偏好集
        # Worker 1-N 是并行采样进程，分配各自的偏好子集
        if worker_index == 0:
            worker_w_set = full_w_batch
        else:
            worker_w_set = w_splits[worker_index - 1]
    else:
        # 本地调试模式（num_workers == 0）
        worker_w_set = full_w_batch
    
    # 将 Worker 专属的偏好集存入 env_config
    env_config["w_batch_set"] = worker_w_set
    
    # 打印详细信息（包含进程ID和偏好范围，避免Ray日志去重）
    pid = os.getpid()
    w_range = f"[{worker_w_set[0][0]:.1f},{worker_w_set[0][1]:.1f}] ~ [{worker_w_set[-1][0]:.1f},{worker_w_set[-1][1]:.1f}]"
    print(f" Worker-{worker_index}/{num_workers} (PID={pid}): 分配 {len(worker_w_set)} 个偏好，范围 {w_range}")
    
    # 创建并返回环境实例
    return DfjsptMaEnv(env_config)


class MyCallbacks(DefaultCallbacks):
    
    def on_episode_start(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        """Episode 开始时的回调 - 用于监控 Worker 活动"""
        # 记录 Worker 信息（用于监控）
        episode.custom_metrics["worker_pid"] = os.getpid()
        episode.custom_metrics["env_index"] = env_index

    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        # if worker.policy_config["batch_mode"] == "truncate_episodes":
        #     # Make sure this episode is really done.
        #     assert episode.batch_builder.policy_collectors["default_policy"].batches[
        #         -1
        #     ]["dones"][-1], (
        #         "ERROR: `on_episode_end()` should only be called "
        #         "after episode is done!"
        #     )
        
        # 记录原有的单目标指标
        episode.custom_metrics["total_makespan"] = episode.worker.env.final_makespan
        episode.custom_metrics["instance_id"] = episode.worker.env.current_instance_id
        episode.custom_metrics["instance_rule_makespan"] = episode.worker.env.rule_makespan_for_current_instance
        episode.custom_metrics["drl_minus_rule"] = episode.worker.env.drl_minus_rule
        
        # 记录多目标指标 (PD-MORL)
        # 从最后一个 agent 的 info 中获取多目标信息
        agents = episode.get_agents()
        if len(agents) > 0:
            last_info = episode.last_info_for(agents[0])
            if last_info is not None and "objectives" in last_info:
                # objectives[0] = -makespan, objectives[1] = -tardiness
                episode.custom_metrics["objectives_makespan"] = last_info["objectives"][0]
                episode.custom_metrics["objectives_tardiness"] = last_info["objectives"][1]
                episode.custom_metrics["total_tardiness"] = last_info.get("total_tardiness", 0.0)
                
                # 记录当前使用的偏好
                if "current_w" in last_info:
                    episode.custom_metrics["preference_w0"] = last_info["current_w"][0]
                    episode.custom_metrics["preference_w1"] = last_info["current_w"][1]
    
    def on_sample_end(
            self,
            *,
            worker: RolloutWorker,
            samples: SampleBatch,
            **kwargs
    ):
        """采样结束时的回调 - 用于监控采样性能"""
        print(f" Worker PID={os.getpid()} 完成采样: {len(samples)} steps")


class MyTrainable(tune.Trainable):
    def setup(self, my_config):
        # self.max_iterations = 500
        self.config = PPOConfig().update_from_dict(my_config)
        self.agent1 = self.config.build()

        self.epoch = 0

    def step(self):
        result = self.agent1.train()
        if result["custom_metrics"]["total_makespan_mean"] <= result["custom_metrics"]["instance_rule_makespan_mean"] - 100:
            dfjspt_params.use_custom_loss = False
        if result["episodes_total"] >= 2e5:
            dfjspt_params.use_custom_loss = False
        self.epoch += 1
        return result

    def save_checkpoint(self, tmp_checkpoint_dir):
        self.agent1.save(tmp_checkpoint_dir)
        # print(f"Checkpoint saved in directory {tmp_checkpoint_dir}")
        return tmp_checkpoint_dir

    # def load_checkpoint(self, tmp_checkpoint_dir):
    #     checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
    #     self.model.load_state_dict(torch.load(checkpoint_path))


if __name__ == "__main__":

    for _ in range(1):

        print(
            f"Start training with {dfjspt_params.n_jobs} jobs, {dfjspt_params.n_machines} machines, and {dfjspt_params.n_transbots} transbots.")

        # 基础 log_dir（基于问题规模）
        base_log_dir = os.path.join(
            os.path.dirname(__file__),
            f"training_results/J{dfjspt_params.n_jobs}_M{dfjspt_params.n_machines}_T{dfjspt_params.n_transbots}"
        )
        os.makedirs(base_log_dir, exist_ok=True)
        
        # 检查是否有环境变量指定的实验名称（由 RUN_EXPERIMENTS.py 设置）
        experiment_name = os.environ.get('DFJSPT_EXPERIMENT_NAME', None)
        
        if experiment_name:
            # 如果有实验名称，创建独立的子目录
            log_dir = os.path.join(base_log_dir, experiment_name)
            print(f"  实验名称: {experiment_name}")
            print(f"  结果保存到: {log_dir}")
        else:
            # 否则使用基础目录（保持向后兼容）
            log_dir = base_log_dir
        
        # 注册带有偏好分配的自定义环境创建器
        tune.register_env("DfjsptMaEnv_PDMORL", create_env_with_preferences)

        example_env = DfjsptMaEnv({
            "train_or_eval_or_test": "train",
        })

        # Define the policies for each agent
        policies = {
            "policy_agent0": (
                None,
                # trained_job_policy,
                example_env.observation_space["agent0"],
                example_env.action_space["agent0"],
                {"model": {
                    "custom_model": DfjsptAgentModel.JOB,
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "tanh",
                }}),
            "policy_agent1": (
                None,
                # trained_machine_policy,
                example_env.observation_space["agent1"],
                example_env.action_space["agent1"],
                {"model": {
                    "custom_model": DfjsptAgentModel.MACHINE,
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "tanh",
                }}),
            "policy_agent2": (
                None,
                # trained_transbot_policy,
                example_env.observation_space["agent2"],
                example_env.action_space["agent2"],
                {"model": {
                    "custom_model": DfjsptAgentModel.TRANSBOT,
                    "fcnet_hiddens": [128, 128],
                        "fcnet_activation": "tanh",
                }}),
        }

        # Define the policy mapping function
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            if agent_id == "agent0":
                return "policy_agent0"
            elif agent_id == "agent1":
                return "policy_agent1"
            else:
                return "policy_agent2"

        num_workers = dfjspt_params.num_workers
        num_gpu = dfjspt_params.num_gpu
        if num_gpu > 0:
            driver_gpu = 0.1
            worker_gpu = (1 - driver_gpu) / num_workers
        else:
            driver_gpu = 0
            worker_gpu = 0
        
        my_config = {
            # environment:
            # 使用新注册的带有偏好分配的环境
            "env": "DfjsptMaEnv_PDMORL",
            "env_config": {
                "train_or_eval_or_test": "train",
                # worker_index 和 num_workers 会由 RLlib 自动注入
            },
            "disable_env_checking": True,
            # framework:
            "framework": dfjspt_params.framework,
            # rollouts:
            "num_rollout_workers": num_workers,
            "num_envs_per_worker": dfjspt_params.num_envs_per_worker,
            "batch_mode": "complete_episodes",
            # debugging：
            "log_level": "WARN",
            "log_sys_usage": True,
            # callbacks：
            "callbacks": MyCallbacks,
            # resources：
            "num_gpus": driver_gpu,
            "num_gpus_per_worker": worker_gpu,
            "num_cpus_per_worker": 1,
            "num_cpus_for_local_worker": 1,
            # evaluation:
            "evaluation_interval": 5,
            "evaluation_duration": 10,
            "evaluation_duration_unit": "episodes",
            "evaluation_parallel_to_training": True,
            "enable_async_evaluation": True,
            "evaluation_num_workers": 1,
            "evaluation_config": PPOConfig.overrides(
                env_config={
                    "train_or_eval_or_test": "eval",
                },
                explore=False,
            ),
            # training:
            "lr_schedule": [
                [0, 3e-5],
                [dfjspt_params.n_jobs * dfjspt_params.n_machines * 5e6, 1e-5]],
            "train_batch_size": dfjspt_params.n_jobs * dfjspt_params.n_machines * max(num_workers, 1) * 120,
            "sgd_minibatch_size": dfjspt_params.n_jobs * dfjspt_params.n_machines * 120,
            "num_sgd_iter": 10,
            "entropy_coeff": 0.001,
            # multi_agent
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        }

        if dfjspt_params.use_her:
            my_config["postprocess_trajectory"] = postprocess_her_trajectory

        stop = {
            "training_iteration": dfjspt_params.stop_iters,
        }

        if not dfjspt_params.use_tune:
            # manual training with train loop using PPO and fixed learning rate
            # if args.run != "PPO":
            #     raise ValueError("Only support --run PPO with --no-tune.")
            print("Running manual train loop without Ray Tune.")

            config = PPOConfig().update_from_dict(my_config)
            algo = config.build()

            # 记录最佳性能
            best_makespan = float('inf')
            best_checkpoint_path = None
            best_iteration = 0
            best_result_info = {}
            
            for i in range(dfjspt_params.stop_iters):
                result = algo.train()
                if result["custom_metrics"]["total_makespan_mean"] <= result["custom_metrics"][
                    "instance_rule_makespan_mean"]:
                    dfjspt_params.use_custom_loss = False
                if result["episodes_total"] >= 1e+5:
                    dfjspt_params.use_custom_loss = False
                
                # 检查是否是最佳性能
                current_makespan = result["custom_metrics"]["total_makespan_mean"]
                if current_makespan < best_makespan:
                    best_makespan = current_makespan
                    best_iteration = i
                    # 保存当前最佳检查点
                    best_checkpoint_path = algo.save()
                    
                    # 保存详细信息
                    best_result_info = {
                        "makespan": current_makespan,
                        "rule_makespan": result["custom_metrics"].get("instance_rule_makespan_mean"),
                        "tardiness": result["custom_metrics"].get("objectives_tardiness_mean"),
                        "total_tardiness": result["custom_metrics"].get("total_tardiness_mean"),
                        "drl_minus_rule": result["custom_metrics"].get("drl_minus_rule_mean"),
                    }
                    
                    print(f" 新的最佳 makespan: {best_makespan:.2f} (iteration {i})")
                    if best_result_info.get("tardiness") is not None:
                        print(f"   Tardiness: {best_result_info['tardiness']:.2f}")
                
                if i % 5 == 0:
                    print(pretty_print(result))
                if i % 20 == 0:
                    checkpoint_dir = algo.save()
                    print(f"Checkpoint saved in directory {checkpoint_dir}")

            checkpoint_dir_end = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir_end}")
            
            # 复制最佳检查点到固定位置
            if best_checkpoint_path is not None:
                best_checkpoint_dir = os.path.join(log_dir, "best_checkpoint")
                
                import shutil
                if os.path.exists(best_checkpoint_dir):
                    shutil.rmtree(best_checkpoint_dir)
                shutil.copytree(best_checkpoint_path, best_checkpoint_dir)
                
                print(f"\n{'='*80}")
                print(f"最佳检查点信息 (手动训练):")
                print(f"  源路径: {best_checkpoint_path}")
                print(f"  最佳 makespan: {best_makespan:.2f}")
                print(f"  规则 makespan: {best_result_info.get('rule_makespan', 'N/A')}")
                if best_result_info.get('tardiness') is not None:
                    print(f"  Tardiness (objectives): {best_result_info['tardiness']:.2f}")
                if best_result_info.get('total_tardiness') is not None:
                    print(f"  Total tardiness: {best_result_info['total_tardiness']:.2f}")
                print(f"  最佳迭代: {best_iteration}")
                print(f"  已复制到: {best_checkpoint_dir}")
                
                # 辅助函数：将 numpy 类型转换为 Python 原生类型
                def convert_to_native_type(value):
                    """将 numpy 类型转换为 JSON 可序列化的 Python 原生类型"""
                    if value is None:
                        return None
                    # 处理 numpy 数值类型
                    if hasattr(value, 'item'):  # numpy 标量
                        return value.item()
                    # 处理 numpy 数组
                    if hasattr(value, 'tolist'):
                        return value.tolist()
                    # 处理字典
                    if isinstance(value, dict):
                        return {k: convert_to_native_type(v) for k, v in value.items()}
                    # 处理列表
                    if isinstance(value, (list, tuple)):
                        return [convert_to_native_type(v) for v in value]
                    # 其他类型直接返回
                    return value
                
                # 保存元信息
                best_info_path = os.path.join(log_dir, "best_checkpoint_info.json")
                best_info = {
                    "checkpoint_path": best_checkpoint_dir,
                    "original_path": best_checkpoint_path,
                    "best_makespan": convert_to_native_type(best_makespan),
                    "rule_makespan": convert_to_native_type(best_result_info.get('rule_makespan')),
                    "objectives_tardiness": convert_to_native_type(best_result_info.get('tardiness')),
                    "total_tardiness": convert_to_native_type(best_result_info.get('total_tardiness')),
                    "drl_minus_rule": convert_to_native_type(best_result_info.get('drl_minus_rule')),
                    "best_iteration": convert_to_native_type(best_iteration),
                    "total_iterations": convert_to_native_type(dfjspt_params.stop_iters),
                }
                
                with open(best_info_path, 'w') as f:
                    json.dump(best_info, f, indent=4)
                
                print(f"  元信息已保存: {best_info_path}")
                print(f"{'='*80}\n")
            
            algo.stop()
        else:
            # automated run with Tune and grid search and TensorBoard
            print("Training automatically with Ray Tune")

            resources = PPO.default_resource_request(my_config)
            tuner = tune.Tuner(
                tune.with_resources(MyTrainable, resources=resources),
                param_space=my_config,
                run_config=air.RunConfig(
                    stop=stop,
                    name=experiment_name,
                    storage_path=os.path.abspath(base_log_dir),
                    checkpoint_config=train.CheckpointConfig(
                        checkpoint_frequency=5, 
                        checkpoint_at_end=True,
                        
                        # # 2. 告知 Tune 如何对检查点进行评分
                        # checkpoint_score_attribute="custom_metrics/total_makespan_mean",
                        
                        # # 3. 告知 Tune 分数越低越好
                        # checkpoint_score_order="min",
                        
                        # 4. 告知 Tune 只保留得分最高的那个检查点
                        num_to_keep=None),
                ),
            )
            results = tuner.fit()
            # print(f"All results: {results}")

            # Get the best result based on a particular metric.
            best_result = results.get_best_result(metric="custom_metrics/total_makespan_mean", mode="min")
            # print(f"Best result: {best_result}")

            # Get the best checkpoint corresponding to the best result.
            best_checkpoint = best_result.checkpoint
            # print(f"Best checkpoint: {best_checkpoint}")
            
            # 保存最佳检查点到固定位置，方便测试使用
            if best_checkpoint is not None:
                # 获取实验目录 - 处理 experiment_name 可能为 None 的情况
                if experiment_name:
                    experiment_dir = os.path.join(base_log_dir, experiment_name)
                else:
                    # 从 best_result 获取实际的实验目录
                    # best_result.log_dir 或 best_result.path 包含完整路径
                    if hasattr(best_result, 'log_dir'):
                        experiment_dir = best_result.log_dir
                    elif hasattr(best_result, 'path'):
                        experiment_dir = os.path.dirname(best_result.path)
                    else:
                        # 从 checkpoint 路径推断
                        if hasattr(best_checkpoint, 'path'):
                            checkpoint_path_str = best_checkpoint.path
                        elif hasattr(best_checkpoint, 'to_directory'):
                            checkpoint_path_str = best_checkpoint.to_directory()
                        else:
                            checkpoint_path_str = str(best_checkpoint)
                        # checkpoint 路径格式: .../experiment_dir/trial_dir/checkpoint_xxx
                        # 我们需要 trial_dir
                        experiment_dir = os.path.dirname(os.path.dirname(checkpoint_path_str))
                
                best_checkpoint_dir = os.path.join(experiment_dir, "best_checkpoint")
                best_info_path = os.path.join(experiment_dir, "best_checkpoint_info.json")
                
                # 从 checkpoint 对象中获取检查点路径
                if hasattr(best_checkpoint, 'path'):
                    checkpoint_path = best_checkpoint.path
                elif hasattr(best_checkpoint, 'to_directory'):
                    # Ray 2.x 新 API
                    checkpoint_path = best_checkpoint.to_directory()
                else:
                    checkpoint_path = str(best_checkpoint)
                
                # 获取指标 - 处理不同的键名格式
                metrics = best_result.metrics
                
                # 尝试不同的键名格式
                makespan_mean = (
                    metrics.get('custom_metrics/total_makespan_mean') or
                    metrics.get('custom_metrics', {}).get('total_makespan_mean') or
                    metrics.get('evaluation', {}).get('custom_metrics', {}).get('total_makespan_mean')
                )
                
                rule_makespan_mean = (
                    metrics.get('custom_metrics/instance_rule_makespan_mean') or
                    metrics.get('custom_metrics', {}).get('instance_rule_makespan_mean') or
                    metrics.get('evaluation', {}).get('custom_metrics', {}).get('instance_rule_makespan_mean')
                )
                
                tardiness_mean = (
                    metrics.get('custom_metrics/objectives_tardiness_mean') or
                    metrics.get('custom_metrics', {}).get('objectives_tardiness_mean') or
                    metrics.get('evaluation', {}).get('custom_metrics', {}).get('objectives_tardiness_mean')
                )
                
                total_tardiness_mean = (
                    metrics.get('custom_metrics/total_tardiness_mean') or
                    metrics.get('custom_metrics', {}).get('total_tardiness_mean') or
                    metrics.get('evaluation', {}).get('custom_metrics', {}).get('total_tardiness_mean')
                )
                
                print(f"\n{'='*80}")
                print(f"最佳检查点信息:")
                print(f"  源路径: {checkpoint_path}")
                print(f"  最佳 makespan: {makespan_mean if makespan_mean is not None else 'N/A'}")
                print(f"  规则 makespan: {rule_makespan_mean if rule_makespan_mean is not None else 'N/A'}")
                print(f"  Tardiness (objectives): {tardiness_mean if tardiness_mean is not None else 'N/A'}")
                print(f"  Total tardiness: {total_tardiness_mean if total_tardiness_mean is not None else 'N/A'}")
                print(f"  训练迭代: {metrics.get('training_iteration', 'N/A')}")
                
                # 复制检查点到固定位置
                import shutil
                if os.path.exists(best_checkpoint_dir):
                    shutil.rmtree(best_checkpoint_dir)
                shutil.copytree(checkpoint_path, best_checkpoint_dir)
                
                print(f"  已复制到: {best_checkpoint_dir}")
                
                # 保存最佳检查点的元信息（使用前面定义的 best_info_path）
                # best_info_path 已在前面定义，这里不需要重复定义
                
                # 辅助函数：将 numpy 类型转换为 Python 原生类型
                def convert_to_native_type(value):
                    """将 numpy 类型转换为 JSON 可序列化的 Python 原生类型"""
                    if value is None:
                        return None
                    # 处理 numpy 数值类型
                    if hasattr(value, 'item'):  # numpy 标量
                        return value.item()
                    # 处理 numpy 数组
                    if hasattr(value, 'tolist'):
                        return value.tolist()
                    # 处理字典
                    if isinstance(value, dict):
                        return {k: convert_to_native_type(v) for k, v in value.items()}
                    # 处理列表
                    if isinstance(value, (list, tuple)):
                        return [convert_to_native_type(v) for v in value]
                    # 其他类型直接返回
                    return value
                
                best_info = {
                    "checkpoint_path": best_checkpoint_dir,
                    "original_path": checkpoint_path,
                    "total_makespan_mean": convert_to_native_type(makespan_mean),
                    "instance_rule_makespan_mean": convert_to_native_type(rule_makespan_mean),
                    "objectives_tardiness_mean": convert_to_native_type(tardiness_mean),
                    "total_tardiness_mean": convert_to_native_type(total_tardiness_mean),
                    "training_iteration": convert_to_native_type(metrics.get('training_iteration')),
                    "episodes_total": convert_to_native_type(metrics.get('episodes_total')),
                    "timestamp": convert_to_native_type(metrics.get('timestamp')),
                    # 保存所有 custom_metrics 以便调试
                    "all_custom_metrics": convert_to_native_type({
                        k: v for k, v in metrics.items() 
                        if k.startswith('custom_metrics')
                    })
                }
                
                with open(best_info_path, 'w') as f:
                    json.dump(best_info, f, indent=4)
                
                # print(f"  元信息已保存: {best_info_path}")
                # print(f"{'='*80}\n")
                
            else:
                print("\n警告: 未找到最佳检查点！")

        ray.shutdown()
