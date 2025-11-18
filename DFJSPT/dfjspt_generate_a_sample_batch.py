import copy
import random
import uuid
import numpy as np
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from DFJSPT.dfjspt_env_for_imitation import DfjsptMaEnv
from DFJSPT.dfjspt_rule.job_selection_rules import job_FDD_MTWR_action
from DFJSPT.dfjspt_rule.machine_selection_rules import machine_EET_action, transbot_EET_action
from ray.rllib.models.preprocessors import get_preprocessor


SOURCES = ("online", "demo", "dagger", "oracle")


def _build_teacher_scores(action_mask, chosen_action):
    """Return teacher scores with -inf on invalid actions.

    For now, valid-but-unselected actions get 0, and the selected action gets 1,
    which works as a soft target while keeping invalid entries masked at -inf.
    """

    scores = np.full_like(action_mask, -np.inf, dtype=np.float32)
    valid_indices = np.where(action_mask > 0)[0]
    scores[valid_indices] = 0.0
    if 0 <= chosen_action < scores.shape[0]:
        scores[chosen_action] = 1.0
    return scores


def generate_sample_batch(batch_type, *, source="demo", pref_vec=None, traj_id=None):
    if source not in SOURCES:
        raise RuntimeError(f"Invalid source type: {source}!")

    job_batch_builder = SampleBatchBuilder()
    machine_batch_builder = SampleBatchBuilder()
    transbot_batch_builder = SampleBatchBuilder()

    env = DfjsptMaEnv()
    job_prep = get_preprocessor(env.observation_space["agent0"])(env.observation_space["agent0"])
    machine_prep = get_preprocessor(env.observation_space["agent1"])(env.observation_space["agent1"])
    transbot_prep = get_preprocessor(env.observation_space["agent2"])(env.observation_space["agent2"])

    observation, info = env.reset()
    if pref_vec is None:
        pref_vec = np.array([0.5, 0.5], dtype=np.float32)
    traj_id = traj_id or str(uuid.uuid4())
    t = 0
    done = False
    count = 0
    stage = next(iter(info["agent0"].values()), None)
    total_reward = 0
    step_id = 0

    while not done:
        if stage == 0:
            job_prev_obs = copy.deepcopy(observation["agent0"])
            legal_job_actions = copy.deepcopy(observation["agent0"]["action_mask"])
            real_job_attrs = copy.deepcopy(observation["agent0"]["observation"])
            FDD_MTWR_job_action = job_FDD_MTWR_action(legal_job_actions=legal_job_actions,
                                                      real_job_attrs=real_job_attrs)
            observation, reward, terminated, truncated, info = env.step(FDD_MTWR_job_action)
            stage = next(iter(info["agent1"].values()), None)
            if batch_type == "job":
                action_mask = np.asarray(job_prev_obs["action_mask"], dtype=np.float32)
                assert action_mask.sum() > 0, "Job action mask must allow at least one action"
                job_batch_builder.add_values(
                    obs_flat=job_prep.transform(job_prev_obs),
                    actions=FDD_MTWR_job_action["agent0"],
                    valid_mask=action_mask,
                    pref_vec=pref_vec,
                    teacher_scores=_build_teacher_scores(action_mask, FDD_MTWR_job_action["agent0"]),
                    source=source,
                    traj_id=traj_id,
                    step_id=step_id,
                )
            t += 1
            step_id += 1

        elif stage == 1:
            machine_prev_obs = copy.deepcopy(observation["agent1"])
            legal_machine_actions = copy.deepcopy(observation["agent1"]["action_mask"])
            real_machine_attrs = copy.deepcopy(observation["agent1"]["observation"])
            EET_machine_action = machine_EET_action(legal_machine_actions=legal_machine_actions,
                                                    real_machine_attrs=real_machine_attrs)
            observation, reward, terminated, truncated, info = env.step(EET_machine_action)
            stage = next(iter(info["agent2"].values()), None)
            if batch_type == "machine":
                action_mask = np.asarray(machine_prev_obs["action_mask"], dtype=np.float32)
                assert action_mask.sum() > 0, "Machine action mask must allow at least one action"
                machine_batch_builder.add_values(
                    obs_flat=machine_prep.transform(machine_prev_obs),
                    actions=EET_machine_action["agent1"],
                    valid_mask=action_mask,
                    pref_vec=pref_vec,
                    teacher_scores=_build_teacher_scores(action_mask, EET_machine_action["agent1"]),
                    source=source,
                    traj_id=traj_id,
                    step_id=step_id,
                )
            t += 1
            step_id += 1
        else:
            transbot_prev_obs = copy.deepcopy(observation["agent2"])
            legal_transbot_actions = copy.deepcopy(observation["agent2"]["action_mask"])
            real_transbot_attrs = copy.deepcopy(observation["agent2"]["observation"])
            EET_transbot_action = transbot_EET_action(legal_transbot_actions=legal_transbot_actions,
                                                      real_transbot_attrs=real_transbot_attrs)
            observation, reward, terminated, truncated, info = env.step(EET_transbot_action)
            stage = next(iter(info["agent0"].values()), None)
            if batch_type == "transbot":
                action_mask = np.asarray(transbot_prev_obs["action_mask"], dtype=np.float32)
                assert action_mask.sum() > 0, "Transbot action mask must allow at least one action"
                transbot_batch_builder.add_values(
                    obs_flat=transbot_prep.transform(transbot_prev_obs),
                    actions=EET_transbot_action["agent2"],
                    valid_mask=action_mask,
                    pref_vec=pref_vec,
                    teacher_scores=_build_teacher_scores(action_mask, EET_transbot_action["agent2"]),
                    source=source,
                    traj_id=traj_id,
                    step_id=step_id,
                )
            t += 1
            step_id += 1

            done = terminated["__all__"]
            count += 1
            total_reward += reward["agent0"]

    if batch_type == "job":
        return job_batch_builder.build_and_reset()
    elif batch_type == "machine":
        return machine_batch_builder.build_and_reset()
    elif batch_type == "transbot":
        return transbot_batch_builder.build_and_reset()
    else:
        raise RuntimeError(f"Invalid batch type: {batch_type}!")


class MultiSourceReplayBuffer:
    def __init__(self, batch_type, *, sample_ratios=None, target_batch_size=120):
        self.batch_type = batch_type
        self.sample_ratios = sample_ratios or {"online": 0.6, "demo": 0.3, "dagger": 0.1}
        self.target_batch_size = target_batch_size
        self.buffers = {source: [] for source in self.sample_ratios}

    def _ensure_seed_data(self):
        for source in self.buffers:
            if not self.buffers[source]:
                self.buffers[source].append(
                    generate_sample_batch(self.batch_type, source=source)
                )

    def add_batch(self, batch: SampleBatch):
        source_values = batch.get("source")
        if source_values is None:
            return
        if isinstance(source_values, (list, np.ndarray)):
            source = source_values[0]
        else:
            source = source_values
        if source not in self.buffers:
            self.buffers[source] = []
        self.buffers[source].append(batch)

    def sample(self):
        self._ensure_seed_data()
        total_weight = sum(self.sample_ratios.values())
        sampled_batches = []
        for source, weight in self.sample_ratios.items():
            target_rows = max(1, int(self.target_batch_size * weight / total_weight))
            pool = self.buffers[source]
            chosen_batch = random.choice(pool)
            available = chosen_batch.count
            if available > target_rows:
                indices = np.random.choice(available, target_rows, replace=False)
                new_data = {k: v[indices] for k, v in chosen_batch.items()}
                sampled_batches.append(SampleBatch(new_data))
            else:
                sampled_batches.append(chosen_batch)
        return concat_samples(sampled_batches)
# my_job_batch = generate_sample_batch("job")
# my_machine_batch = generate_sample_batch("machine")
# my_transbot_batch = generate_sample_batch("transbot")
