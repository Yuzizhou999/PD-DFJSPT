from gymnasium.spaces import Dict, Box
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from DFJSPT import dfjspt_params
from DFJSPT.dfjspt_generate_a_sample_batch import (
    MultiSourceReplayBuffer,
    generate_sample_batch,
)

torch, nn = try_import_torch()


JOB_REPLAY = MultiSourceReplayBuffer("job")
MACHINE_REPLAY = MultiSourceReplayBuffer("machine")
TRANSBOT_REPLAY = MultiSourceReplayBuffer("transbot")


class JobActionMaskModel(TorchModelV2, nn.Module):
    """
    PyTorch version of ActionMaskingModel with Preference-Driven Multi-Objective support.
    
    This model accepts observations that include both the original features and preference vectors.
    The preference is concatenated with the observation before being fed to the network.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observation" in orig_space.spaces
            and "preference" in orig_space.spaces  # PD-MORL: 确认 preference 存在
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        # PD-MORL: 获取原始观测和偏好的维度
        self.obs_shape = orig_space["observation"].shape  # (MAX_JOBS, n_features)
        self.preference_dim = orig_space["preference"].shape[0]  # reward_size (通常是 2)
        
        # PD-MORL: 为 imitation learning 保存不含偏好的原始观测空间
        self.orig_obs_space_for_il = Dict({
            "observation": orig_space["observation"],
            "action_mask": orig_space["action_mask"]
        })
        
        # PD-MORL: 计算展平后的观测维度
        # observation 是二维的 (MAX_JOBS, n_features)，需要展平
        self.obs_dim = 1
        for dim in self.obs_shape:
            self.obs_dim *= dim
        
        # PD-MORL: 创建组合空间（observation + preference）
        # 新的输入维度 = 展平的观测维度 + 偏好维度
        combined_dim = self.obs_dim + self.preference_dim
        
        # 创建一个临时的 Box 空间来初始化 internal_model
        from gymnasium.spaces import Box
        combined_space = Box(
            low=-float('inf'), 
            high=float('inf'), 
            shape=(combined_dim,),
            dtype=orig_space["observation"].dtype
        )
        
        self.internal_model = TorchFC(
            combined_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        
        # PD-MORL: 提取观测和偏好
        observation = input_dict["obs"]["observation"]
        preference = input_dict["obs"]["preference"]
        
        # PD-MORL: 展平观测 (batch_size, MAX_JOBS, n_features) -> (batch_size, obs_dim)
        batch_size = observation.shape[0]
        observation_flat = observation.reshape(batch_size, -1)
        
        # PD-MORL: 拼接观测和偏好
        # observation_flat: (batch_size, obs_dim)
        # preference: (batch_size, preference_dim)
        # combined: (batch_size, obs_dim + preference_dim)
        combined_input = torch.cat([observation_flat, preference], dim=1)

        # Compute the unmasked logits using the combined input
        logits, _ = self.internal_model({"obs": combined_input})

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    @override(ModelV2)
    def value_function(self):
        return self.internal_model.value_function()

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs=None):
        if dfjspt_params.use_custom_loss is True:
            # Sample a mixed batch from the replay pools.
            batch = JOB_REPLAY.sample()

            # PD-MORL: 使用不含偏好的原始观测空间进行维度还原
            obs = restore_original_dimensions(
                torch.from_numpy(batch["obs_flat"]).float().to(policy_loss[0].device),
                self.orig_obs_space_for_il,  # 使用不含偏好的空间
                tensorlib="torch",
            )

            # 使用轨迹自带的偏好向量，保证同一 traj_id 的偏好一致
            pref_tensor = torch.from_numpy(batch["pref_vec"]).float().to(policy_loss[0].device)
            obs["preference"] = pref_tensor

            logits, _ = self.forward({"obs": obs}, [], None)

            action_mask = torch.from_numpy(batch["valid_mask"]).to(policy_loss[0].device)
            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
            logits = logits + inf_mask

            # Compute the IL loss.
            action_dist_1 = TorchCategorical(logits, self.model_config)

            imitation_loss_1 = torch.mean(
                -action_dist_1.logp(
                    # torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                    torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                )
            )
            imitation_loss = imitation_loss_1
            # self.imitation_loss_metric = imitation_loss.item()
            # self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])

            # Add the imitation loss to each already calculated policy loss term.
            # Alternatively (if custom loss has its own optimizer):
            # return policy_loss + [10 * self.imitation_loss]
            return [loss_ + dfjspt_params.il_loss_weight * imitation_loss for loss_ in policy_loss]
        elif dfjspt_params.use_custom_loss is False:
            return policy_loss
        else:
            raise RuntimeError('Invalid "use_custom_loss" value!')



class MachineActionMaskModel(TorchModelV2, nn.Module):
    """
    PyTorch version of ActionMaskingModel with Preference-Driven Multi-Objective support.
    
    This model accepts observations that include both the original features and preference vectors.
    The preference is concatenated with the observation before being fed to the network.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observation" in orig_space.spaces
            and "preference" in orig_space.spaces  # PD-MORL: 确认 preference 存在
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        # PD-MORL: 获取原始观测和偏好的维度
        self.obs_shape = orig_space["observation"].shape  # (MAX_MACHINES, n_features)
        self.preference_dim = orig_space["preference"].shape[0]  # reward_size
        
        # PD-MORL: 为 imitation learning 保存不含偏好的原始观测空间
        self.orig_obs_space_for_il = Dict({
            "observation": orig_space["observation"],
            "action_mask": orig_space["action_mask"]
        })
        
        # PD-MORL: 计算展平后的观测维度
        self.obs_dim = 1
        for dim in self.obs_shape:
            self.obs_dim *= dim
        
        # PD-MORL: 创建组合空间（observation + preference）
        combined_dim = self.obs_dim + self.preference_dim
        
        from gymnasium.spaces import Box
        combined_space = Box(
            low=-float('inf'), 
            high=float('inf'), 
            shape=(combined_dim,),
            dtype=orig_space["observation"].dtype
        )
        
        self.internal_model = TorchFC(
            combined_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        
        # PD-MORL: 提取观测和偏好
        observation = input_dict["obs"]["observation"]
        preference = input_dict["obs"]["preference"]
        
        # PD-MORL: 展平观测
        batch_size = observation.shape[0]
        observation_flat = observation.reshape(batch_size, -1)
        
        # PD-MORL: 拼接观测和偏好
        combined_input = torch.cat([observation_flat, preference], dim=1)

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": combined_input})

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs=None):
        if dfjspt_params.use_custom_loss is True:
            # Sample a mixed batch from the replay pools.
            batch = MACHINE_REPLAY.sample()

            # PD-MORL: 使用不含偏好的原始观测空间进行维度还原
            obs = restore_original_dimensions(
                torch.from_numpy(batch["obs_flat"]).float().to(policy_loss[0].device),
                self.orig_obs_space_for_il,  # 使用不含偏好的空间
                tensorlib="torch",
            )
            
            # 使用轨迹自带的偏好向量，保证同一 traj_id 的偏好一致
            pref_tensor = torch.from_numpy(batch["pref_vec"]).float().to(policy_loss[0].device)
            obs["preference"] = pref_tensor

            logits, _ = self.forward({"obs": obs}, [], None)
            action_mask = torch.from_numpy(batch["valid_mask"]).to(policy_loss[0].device)
            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
            logits = logits + inf_mask

            # Compute the IL loss.
            action_dist_1 = TorchCategorical(logits, self.model_config)

            imitation_loss_1 = torch.mean(
                -action_dist_1.logp(
                    # torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                    torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                )
            )
            imitation_loss = imitation_loss_1
            # self.imitation_loss_metric = imitation_loss.item()
            # self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])

            # Add the imitation loss to each already calculated policy loss term.
            # Alternatively (if custom loss has its own optimizer):
            # return policy_loss + [10 * self.imitation_loss]
            return [loss_ + dfjspt_params.il_loss_weight * imitation_loss for loss_ in policy_loss]
        else:
            return policy_loss


class TransbotActionMaskModel(TorchModelV2, nn.Module):
    """
    PyTorch version of ActionMaskingModel with Preference-Driven Multi-Objective support.
    
    This model accepts observations that include both the original features and preference vectors.
    The preference is concatenated with the observation before being fed to the network.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, Dict)
                and "action_mask" in orig_space.spaces
                and "observation" in orig_space.spaces
                and "preference" in orig_space.spaces  # PD-MORL: 确认 preference 存在
        )
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        # PD-MORL: 获取原始观测和偏好的维度
        self.obs_shape = orig_space["observation"].shape  # (MAX_TRANSBOTS, n_features)
        self.preference_dim = orig_space["preference"].shape[0]  # reward_size
        
        # PD-MORL: 为 imitation learning 保存不含偏好的原始观测空间
        self.orig_obs_space_for_il = Dict({
            "observation": orig_space["observation"],
            "action_mask": orig_space["action_mask"]
        })
        
        # PD-MORL: 计算展平后的观测维度
        self.obs_dim = 1
        for dim in self.obs_shape:
            self.obs_dim *= dim
        
        # PD-MORL: 创建组合空间（observation + preference）
        combined_dim = self.obs_dim + self.preference_dim
        
        from gymnasium.spaces import Box
        combined_space = Box(
            low=-float('inf'), 
            high=float('inf'), 
            shape=(combined_dim,),
            dtype=orig_space["observation"].dtype
        )
        
        self.internal_model = TorchFC(
            combined_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]
        
        # PD-MORL: 提取观测和偏好
        observation = input_dict["obs"]["observation"]
        preference = input_dict["obs"]["preference"]
        
        # PD-MORL: 展平观测
        batch_size = observation.shape[0]
        observation_flat = observation.reshape(batch_size, -1)
        
        # PD-MORL: 拼接观测和偏好
        combined_input = torch.cat([observation_flat, preference], dim=1)

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": combined_input})

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    @override(ModelV2)
    def value_function(self):
        return self.internal_model.value_function()

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs=None):
        if dfjspt_params.use_custom_loss is True:
            # Sample a mixed batch from the replay pools.
            batch = TRANSBOT_REPLAY.sample()

            # PD-MORL: 使用不含偏好的原始观测空间进行维度还原
            obs = restore_original_dimensions(
                torch.from_numpy(batch["obs_flat"]).float().to(policy_loss[0].device),
                self.orig_obs_space_for_il,  # 使用不含偏好的空间
                tensorlib="torch",
            )
            
            # 使用轨迹自带的偏好向量，保证同一 traj_id 的偏好一致
            pref_tensor = torch.from_numpy(batch["pref_vec"]).float().to(policy_loss[0].device)
            obs["preference"] = pref_tensor

            logits, _ = self.forward({"obs": obs}, [], None)
            action_mask = torch.from_numpy(batch["valid_mask"]).to(policy_loss[0].device)
            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
            logits = logits + inf_mask

            # Compute the IL loss.
            action_dist = TorchCategorical(logits, self.model_config)

            imitation_loss = torch.mean(
                -action_dist.logp(
                    # torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                    torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
                )
            )
            # self.imitation_loss_metric = imitation_loss.item()
            # self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])

            # Add the imitation loss to each already calculated policy loss term.
            # Alternatively (if custom loss has its own optimizer):
            # return policy_loss + [10 * self.imitation_loss]
            return [loss_ + dfjspt_params.il_loss_weight * imitation_loss for loss_ in policy_loss]
        elif dfjspt_params.use_custom_loss is False:
            return policy_loss
        else:
            raise RuntimeError('Invalid "use_custom_loss" value!')
