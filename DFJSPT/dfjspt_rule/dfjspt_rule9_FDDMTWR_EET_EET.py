import copy

from matplotlib import pyplot as plt

import numpy as np
import time
import json
from DFJSPT import dfjspt_params
from DFJSPT.dfjspt_rule.job_selection_rules import job_FDD_MTWR_action
from DFJSPT.dfjspt_rule.machine_selection_rules import machine_EET_action, transbot_EET_action
from DFJSPT.env_for_rule import DfjsptMaEnv_for_rule

def rule9_mean_makespan(
    instance_id,
    train_or_eval_or_test=None,
):

    config = {
        "train_or_eval_or_test": train_or_eval_or_test,
    }
    env = DfjsptMaEnv_for_rule(config)
    makespan_list = []
    for _ in range(100):
        observation, info = env.reset(
            # options={"instance_id": instance_id,}
        )
        # env.render()
        done = False
        count = 0
        stage = next(iter(info["agent0"].values()), None)
        total_reward = 0

        while not done:
            if stage == 0:
                legal_job_actions = copy.deepcopy(observation["agent0"]["action_mask"])
                real_job_attrs = copy.deepcopy(observation["agent0"]["observation"])
                FDD_MTWR_job_action = job_FDD_MTWR_action(legal_job_actions=legal_job_actions,
                                                          real_job_attrs=real_job_attrs)
                observation, reward, terminated, truncated, info = env.step(FDD_MTWR_job_action)
                stage = next(iter(info["agent1"].values()), None)

            elif stage == 1:
                legal_machine_actions = copy.deepcopy(observation["agent1"]["action_mask"])
                real_machine_attrs = copy.deepcopy(observation["agent1"]["observation"])
                EET_machine_action = machine_EET_action(legal_machine_actions=legal_machine_actions,
                                                        real_machine_attrs=real_machine_attrs)
                observation, reward, terminated, truncated, info = env.step(EET_machine_action)
                stage = next(iter(info["agent2"].values()), None)

            else:
                legal_transbot_actions = copy.deepcopy(observation["agent2"]["action_mask"])
                real_transbot_attrs = copy.deepcopy(observation["agent2"]["observation"])
                EET_transbot_action = transbot_EET_action(legal_transbot_actions=legal_transbot_actions,
                                                          real_transbot_attrs=real_transbot_attrs)
                observation, reward, terminated, truncated, info = env.step(EET_transbot_action)
                stage = next(iter(info["agent0"].values()), None)
                done = terminated["__all__"]
                count += 1
                total_reward += reward["agent0"]

        make_span = env.final_makespan
        # env.render()
        # time.sleep(15)
        makespan_list.append(make_span)
    mean_makespan = np.mean(makespan_list)
    return makespan_list, mean_makespan


def rule9_single_makespan(
    instance_id,
    train_or_eval_or_test=None,
):

    config = {
        "train_or_eval_or_test": train_or_eval_or_test,
    }
    env = DfjsptMaEnv_for_rule(config)

    observation, info = env.reset(options={
        "instance_id": instance_id,

    })
    # env.render()
    done = False
    count = 0
    stage = next(iter(info["agent0"].values()), None)
    total_reward = 0

    while not done:
        if stage == 0:
            legal_job_actions = copy.deepcopy(observation["agent0"]["action_mask"])
            real_job_attrs = copy.deepcopy(observation["agent0"]["observation"])

            FDD_MTWR_job_action = job_FDD_MTWR_action(legal_job_actions=legal_job_actions,
                                                      real_job_attrs=real_job_attrs)
            observation, reward, terminated, truncated, info = env.step(FDD_MTWR_job_action)
            stage = next(iter(info["agent1"].values()), None)

        elif stage == 1:
            legal_machine_actions = copy.deepcopy(observation["agent1"]["action_mask"])
            real_machine_attrs = copy.deepcopy(observation["agent1"]["observation"])

            EET_machine_action = machine_EET_action(legal_machine_actions=legal_machine_actions,
                                                    real_machine_attrs=real_machine_attrs)
            observation, reward, terminated, truncated, info = env.step(EET_machine_action)
            stage = next(iter(info["agent2"].values()), None)

        else:
            legal_transbot_actions = copy.deepcopy(observation["agent2"]["action_mask"])
            real_transbot_attrs = copy.deepcopy(observation["agent2"]["observation"])

            EET_transbot_action = transbot_EET_action(legal_transbot_actions=legal_transbot_actions, real_transbot_attrs=real_transbot_attrs)
            observation, reward, terminated, truncated, info = env.step(EET_transbot_action)
            stage = next(iter(info["agent0"].values()), None)
            done = terminated["__all__"]
            count += 1
            total_reward += reward["agent0"]

    make_span = env.final_makespan

    return make_span

def rule9_a_makespan(
    instance,
):

    config = {
        "instance": instance,
    }
    env = DfjsptMaEnv_for_rule(config)

    observation, info = env.reset()
    # env.render()
    done = False
    count = 0
    stage = next(iter(info["agent0"].values()), None)
    total_reward = 0

    while not done:
        if stage == 0:
            legal_job_actions = copy.deepcopy(observation["agent0"]["action_mask"])
            real_job_attrs = copy.deepcopy(observation["agent0"]["observation"])

            FDD_MTWR_job_action = job_FDD_MTWR_action(legal_job_actions=legal_job_actions,
                                                      real_job_attrs=real_job_attrs)
            observation, reward, terminated, truncated, info = env.step(FDD_MTWR_job_action)
            stage = next(iter(info["agent1"].values()), None)

        elif stage == 1:
            legal_machine_actions = copy.deepcopy(observation["agent1"]["action_mask"])
            real_machine_attrs = copy.deepcopy(observation["agent1"]["observation"])

            EET_machine_action = machine_EET_action(legal_machine_actions=legal_machine_actions,
                                                    real_machine_attrs=real_machine_attrs)
            observation, reward, terminated, truncated, info = env.step(EET_machine_action)
            stage = next(iter(info["agent2"].values()), None)

        else:
            legal_transbot_actions = copy.deepcopy(observation["agent2"]["action_mask"])
            real_transbot_attrs = copy.deepcopy(observation["agent2"]["observation"])

            EET_transbot_action = transbot_EET_action(legal_transbot_actions=legal_transbot_actions, real_transbot_attrs=real_transbot_attrs)
            observation, reward, terminated, truncated, info = env.step(EET_transbot_action)
            stage = next(iter(info["agent0"].values()), None)
            done = terminated["__all__"]
            count += 1
            total_reward += reward["agent0"]

    make_span = env.final_makespan
    total_tardiness = env.get_total_tardiness()

    return make_span, total_tardiness


if __name__ == '__main__':
    import os
    folder_name = os.path.dirname(os.path.dirname(__file__)) + "/dfjspt_data/my_data/pool_J" + str(dfjspt_params.n_jobs) + "_M" + str(dfjspt_params.n_machines) + "_T" + str(dfjspt_params.n_transbots)

    makespan_list = []
    complexity_list = []
    for j in range(dfjspt_params.n_instances_for_testing):

        _, makespan = rule9_mean_makespan(
            instance_id=j,
            train_or_eval_or_test="test",
        )
        makespan_list.append(makespan)
    average_makespan = np.mean(makespan_list)
    makespan_sorted_indices = [makespan_list.index(x) for x in sorted(makespan_list)]
    print(makespan_list)
    print(average_makespan)



