import json
import os
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
from ray.rllib.policy.sample_batch import SampleBatch
# from ray.rllib.algorithms.algorithm import Algorithm
from typing import Dict
# from gymnasium import spaces
# import torch
# import numpy as np

from DFJSPT import dfjspt_params
from DFJSPT.dfjspt_env import DfjsptMaEnv
from DFJSPT.dfjspt_agent_model import JobActionMaskModel, MachineActionMaskModel, TransbotActionMaskModel


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
    W_STEP_SIZE = 0.1
    
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
    print(f"🔧 Worker-{worker_index}/{num_workers} (PID={pid}): 分配 {len(worker_w_set)} 个偏好，范围 {w_range}")
    
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
        print(f"✅ Worker PID={os.getpid()} 完成采样: {len(samples)} steps")


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

        log_dir = os.path.dirname(__file__) + "/training_results/J" + str(
            dfjspt_params.n_jobs) + "_M" + str(dfjspt_params.n_machines) + "_T" + str(dfjspt_params.n_transbots)

        # 注册自定义模型
        ModelCatalog.register_custom_model(
            "job_agent_model", JobActionMaskModel
        )
        ModelCatalog.register_custom_model(
            "machine_agent_model", MachineActionMaskModel
        )
        ModelCatalog.register_custom_model(
            "transbot_agent_model", TransbotActionMaskModel
        )
        
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
                    "custom_model": "job_agent_model",
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "tanh",
                }}),
            "policy_agent1": (
                None,
                # trained_machine_policy,
                example_env.observation_space["agent1"],
                example_env.action_space["agent1"],
                {"model": {
                    "custom_model": "machine_agent_model",
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "tanh",
                }}),
            "policy_agent2": (
                None,
                # trained_transbot_policy,
                example_env.observation_space["agent2"],
                example_env.action_space["agent2"],
                {"model": {
                    "custom_model": "transbot_agent_model",
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

            for i in range(dfjspt_params.stop_iters):
                result = algo.train()
                if result["custom_metrics"]["total_makespan_mean"] <= result["custom_metrics"][
                    "instance_rule_makespan_mean"]:
                    dfjspt_params.use_custom_loss = False
                if result["episodes_total"] >= 1e+5:
                    dfjspt_params.use_custom_loss = False
                if i % 5 == 0:
                    print(pretty_print(result))
                if i % 20 == 0:
                    checkpoint_dir = algo.save()
                    print(f"Checkpoint saved in directory {checkpoint_dir}")

            checkpoint_dir_end = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir_end}")
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
                    name=log_dir,
                    checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10, checkpoint_at_end=True),
                ),
            )
            results = tuner.fit()

            # Get the best result based on a particular metric.
            best_result = results.get_best_result(metric="custom_metrics/total_makespan_mean", mode="min")
            print(best_result)

            # Get the best checkpoint corresponding to the best result.
            best_checkpoint = best_result.checkpoint
            print(best_checkpoint)

        ray.shutdown()



