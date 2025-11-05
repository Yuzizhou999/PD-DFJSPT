from typing import List

# from memory_profiler import profile
# @profile
# def func():
#     print("2")

from gymnasium.spaces import Box, Discrete, Dict
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import matplotlib as mpl
import networkx as nx
import pandas as pd
import random
import cv2
from DFJSPT import dfjspt_params
from DFJSPT.dfjspt_data.dfjspt_data_generator import generate_a_complete_instance
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# from ray.rllib.utils.spaces.repeated import Repeated

# Constraints on the Repeated space.
MAX_JOBS = dfjspt_params.max_n_jobs
MAX_MACHINES = dfjspt_params.max_n_machines
MAX_TRANSBOTS = dfjspt_params.max_n_transbots


# ENV_COUNT = 0

# A Multi-agent Environment for Dynamic Flexible Job-shop Scheduling Problem with Transportation
class DfjsptMaEnv_for_rule(MultiAgentEnv):

    def __init__(self, env_config):
        super().__init__()

        # ------ instance size parameters ------ #
        self.n_jobs = None
        self.n_machines = None
        self.n_transbots = None
        self.max_n_operations = None
        self.min_n_operations = None
        self.n_total_tasks = None
        self.n_processing_tasks = None
        self.n_total_nodes = None
        self.n_job_features = 8
        self.n_machine_features = 7
        self.n_transbot_features = 7

        self.instance = env_config["instance"]
        del env_config

        self.Graph = None
        # self.instance = None
        self.source_task = -1
        # self.sink_task = self.n_total_tasks - 1
        self.sink_task = None

        self.reward_this_step = None
        self.stage = None
        self.chosen_job = -1
        self.chosen_machine = -1
        self.chosen_transbot = -1
        self.operation_id = -1
        self.tspt_task_id = -1
        self.prcs_task_id = -1
        self.perform_left_shift_if_possible = dfjspt_params.perform_left_shift_if_possible
        self.schedule_done = False

        self.job_features = None
        self.machine_features = None
        self.transbot_features = None
        self.job_action_mask = None
        self.machine_action_mask = None
        self.transbot_action_mask = None
        self.env_current_time = 0
        self.final_makespan = 0
        self.drl_minus_rule = 0
        self.prev_cmax = None
        self.curr_cmax = None
        self.result_start_time_for_jobs = None
        self.result_finish_time_for_jobs = None
        self.current_instance_id = 0
        self.instance_count = 0

        # 'routes' of the machines. indicates in which order a machine processes tasks
        self.machine_routes = None
        # 'routes' of the transbots. indicates in which order a machine processes tasks
        self.transbot_routes = None

        # agent0: operation selection agent
        # agent1: machine selection agent
        # agent2: transbot selection agent
        self.agents = {"agent0", "agent1", "agent2"}
        self._agent_ids = set(self.agents)
        self.terminateds = set()
        self.truncateds = set()

        self.resetted = False

    def _get_info(self):
        if self.stage == 0:
            return {
                "agent0": {
                    "current_stage": self.stage
                },
            }
        elif self.stage == 1:
            return {
                "agent1": {
                    "current_stage": self.stage
                },
            }
        else:
            return {
                "agent2": {
                    "current_stage": self.stage
                },
            }

    def _get_obs(self):
        if self.stage == 0:
            return {
                "agent0": {
                    "action_mask": self.job_action_mask,
                    "observation": self.job_features,
                },
            }
        elif self.stage == 1:
            return {
                "agent1": {
                    "action_mask": self.machine_action_mask,
                    "observation": self.machine_features,
                },
            }
        else:
            return {
                "agent2": {
                    "action_mask": self.transbot_action_mask,
                    "observation": self.transbot_features,
                }
            }

    def reset(self, seed=None, options=None):
        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()

        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # self.instance = random.choice(self.my_instance_pool)

        self.n_operations_for_jobs = np.array(self.instance[0])
        self.job_arrival_time = np.array(self.instance[1])
        self.processing_time_matrix = np.array(self.instance[2])
        self.empty_moving_time_matrix = np.array(self.instance[3])
        self.transport_time_matrix = np.array(self.instance[4])
        self.machine_quality = np.array(self.instance[5])
        self.transbot_quality = np.array(self.instance[6])
        del self.instance

        # self.instance[0]: n_operations_for_jobs
        # self.instance[1]: processing_time_baseline
        # self.instance[2]: processing_time_matrix
        # self.instance[3]: empty_moving_time_matrix
        # self.instance[4]: transport_time_matrix
        # self.instance[5]: machine_quality
        # self.instance[6]: transbot_quality

        self.env_current_time = 0
        self.n_jobs = len(self.n_operations_for_jobs)
        self.max_n_operations = np.max(self.n_operations_for_jobs)
        self.min_n_operations = np.min(self.n_operations_for_jobs)
        self.n_machines = len(self.machine_quality[0])
        self.n_transbots = len(self.transbot_quality[0])
        self.n_total_tasks = np.sum(self.n_operations_for_jobs) + 2
        self.n_total_nodes = self.n_total_tasks + self.n_machines + self.n_transbots
        self.sink_task = self.n_total_tasks - 2

        # reset start_time and finish_time.
        self.result_start_time_for_jobs = np.zeros(shape=(self.n_jobs, self.max_n_operations, 2), dtype=float)
        self.result_finish_time_for_jobs = np.zeros(shape=(self.n_jobs, self.max_n_operations, 2), dtype=float)
        self.mean_processing_time_of_operations = np.zeros(shape=(self.n_jobs, self.max_n_operations), dtype=float)
        for job_id in range(self.n_jobs):
            for operation_id in range(self.n_operations_for_jobs[job_id]):
                processing_time_of_operations = self.processing_time_matrix[job_id][operation_id][
                    np.where(self.processing_time_matrix[job_id][operation_id] > 0)]
                # self.machine_mask[job_id][operation_id] = \
                #     [0 if value <= 0 else 1 for value in self.processing_time_matrix[job_id][operation_id]]
                self.mean_processing_time_of_operations[job_id][operation_id] = np.mean(processing_time_of_operations)
        self.mean_cumulative_processing_time_of_jobs = np.cumsum(self.mean_processing_time_of_operations, axis=1)

        self.rule_makespan_for_current_instance = np.max(self.mean_cumulative_processing_time_of_jobs)

        self.initialize_disjunctive_graph(self.n_operations_for_jobs)

        # reset machine routes dict and transbot routes dict
        self.machine_routes = {m_id: np.array([], dtype=int)
                               for m_id in range(self.n_machines)}
        self.transbot_routes = {t_id: np.array([], dtype=int)
                                for t_id in range(self.n_transbots)}

        # remove is_scheduled flags and is_valid flag.
        self.job_features = np.zeros((MAX_JOBS, self.n_job_features), dtype=float)
        for job_id in range(self.n_jobs):
            self.job_features[job_id, 0] = job_id
        self.job_features[:self.n_jobs, 2] = self.job_arrival_time
        self.job_features[:, 3] = self.n_machines
        self.job_features[self.n_jobs:, 4] = 1
        self.job_features[:self.n_jobs, 6] = self.mean_processing_time_of_operations[:, 0]
        self.job_features[:self.n_jobs, 7] = self.mean_cumulative_processing_time_of_jobs[:, -1]

        self.machine_features = np.zeros((MAX_MACHINES, self.n_machine_features), dtype=float)
        for machine_id in range(self.n_machines):
            self.machine_features[machine_id, 0] = machine_id
        self.machine_features[:self.n_machines, 1] = self.machine_quality
        self.machine_features[:self.n_machines, 5:] = -1

        self.transbot_features = np.zeros((MAX_TRANSBOTS, self.n_transbot_features), dtype=float)
        for transbot_id in range(self.n_transbots):
            self.transbot_features[transbot_id, 0] = transbot_id
        self.transbot_features[:self.n_transbots, 1] = self.transbot_quality
        self.transbot_features[:self.n_transbots, 4] = self.n_machines
        self.transbot_features[:self.n_transbots, 6] = -1

        # Initial Action mask
        self.job_action_mask = np.zeros((MAX_JOBS,), dtype=int)
        for job_id in range(self.n_jobs):
            if self.job_arrival_time[job_id] == 0:
                self.job_action_mask[job_id] = 1

        self.machine_action_mask = np.zeros((MAX_MACHINES,), dtype=int)
        self.machine_action_mask[:self.n_machines] = 1

        self.transbot_action_mask = np.zeros((MAX_TRANSBOTS,), dtype=int)
        self.transbot_action_mask[:self.n_transbots] = 1

        # Generate due dates for each job (for Total Tardiness calculation)
        # Due date is based on arrival time + a multiplier of the mean cumulative processing time
        total_proc_times = self.mean_cumulative_processing_time_of_jobs[:, -1]
        due_date_factor = 1.5  # Same as in dfjspt_env.py
        self.job_due_dates = self.job_arrival_time + (total_proc_times * due_date_factor)

        self.prev_cmax = 0
        self.curr_cmax = 0
        self.reward_this_step = 0.0
        self.schedule_done = False
        self.stage = 0
        observations = self._get_obs()
        info = self._get_info()

        return observations, info

    def get_total_tardiness(self):
        """
        Calculate the total tardiness across all jobs.
        Tardiness = max(0, completion_time - due_date)
        
        Returns:
            total_tardiness (float): Sum of tardiness across all jobs
        """
        total_tardiness = 0.0
        for job_id in range(self.n_jobs):
            due_date = self.job_due_dates[job_id]
            
            # Check if job is completed
            if self.job_features[job_id, 4] == 1:
                # Job completed - use actual completion time
                last_operation_id = self.n_operations_for_jobs[job_id] - 1
                completion_time = self.result_finish_time_for_jobs[job_id, last_operation_id, 1]
            else:
                # Job not completed - estimate completion time
                current_time = float(self.job_features[job_id, 2])
                remaining_time = float(self.job_features[job_id, 7])
                completion_time = current_time + remaining_time
            
            # Calculate tardiness
            tardiness = max(0.0, completion_time - due_date)
            total_tardiness += tardiness
        
        return total_tardiness

    def step(self, action):
        observations, reward, terminated, truncated, info = {}, {}, {}, {}, {}
        self.reward_this_step = 0.0

        if self.stage == 0:
            self.chosen_job = action["agent0"]

            # invalid job_id
            if self.chosen_job >= self.n_jobs or self.chosen_job < 0:
                self.schedule_done = min(self.job_features[:, 4]) >= 1
                if self.schedule_done:
                    self.final_makespan = self.curr_cmax
                    self.reward_this_step = self.reward_this_step + 1
                for i in range(3):
                    reward["agent{}".format(i)] = self.reward_this_step
                    terminated["agent{}".format(i)] = self.schedule_done
                    if terminated["agent{}".format(i)]:
                        self.terminateds.add("agent{}".format(i))
                    truncated["agent{}".format(i)] = False
                observations = self._get_obs()
                info = self._get_info()
                terminated["__all__"] = len(self.terminateds) == len(self.agents)
                truncated["__all__"] = False
                return observations, reward, terminated, truncated, info

            self.operation_id = int(self.job_features[self.chosen_job, 1])
            self.tspt_task_id = sum(self.n_operations_for_jobs[:self.chosen_job]) + self.operation_id
            self.prcs_task_id = self.tspt_task_id

            # invalid operation_id
            if self.operation_id >= self.n_operations_for_jobs[self.chosen_job] or self.operation_id < 0:
                self.schedule_done = min(self.job_features[:, 4]) >= 1
                if self.schedule_done:
                    self.final_makespan = self.curr_cmax
                    self.reward_this_step = self.reward_this_step + 1
                for i in range(3):
                    reward["agent{}".format(i)] = self.reward_this_step
                    terminated["agent{}".format(i)] = self.schedule_done
                    if terminated["agent{}".format(i)]:
                        self.terminateds.add("agent{}".format(i))
                    truncated["agent{}".format(i)] = False
                observations = self._get_obs()
                info = self._get_info()
                terminated["__all__"] = len(self.terminateds) == len(self.agents)
                truncated["__all__"] = False
                return observations, reward, terminated, truncated, info

            # update machines' state
            for machine_id in range(self.n_machines):
                # The expected time required for this machine to process the chosen job
                self.machine_features[machine_id, 5] = max(-1.0, self.processing_time_matrix[
                    self.chosen_job, self.operation_id, machine_id] / self.machine_features[machine_id, 1])
                # The expected time to transport the chosen job to this machine
                self.machine_features[machine_id, 6] = self.transport_time_matrix[
                    int(self.job_features[self.chosen_job, 3]),
                    int(self.machine_features[machine_id, 0])]
                if self.machine_features[machine_id, 5] > 0:
                    self.machine_action_mask[machine_id] = 1
                else:
                    self.machine_action_mask[machine_id] = 0

            self.stage = 1

        elif self.stage == 1:
            self.chosen_machine = action["agent1"]

            # invalid machine_id
            if self.chosen_machine >= self.n_machines or self.chosen_machine < 0 or self.machine_features[
                self.chosen_machine, 5] <= 0:
                self.schedule_done = min(self.job_features[:, 4]) >= 1
                if self.schedule_done:
                    self.final_makespan = self.curr_cmax
                    self.reward_this_step = self.reward_this_step + 1
                for i in range(3):
                    reward["agent{}".format(i)] = self.reward_this_step
                    terminated["agent{}".format(i)] = self.schedule_done
                    if terminated["agent{}".format(i)]:
                        self.terminateds.add("agent{}".format(i))
                    truncated["agent{}".format(i)] = False
                observations = self._get_obs()
                info = self._get_info()
                terminated["__all__"] = len(self.terminateds) == len(self.agents)
                truncated["__all__"] = False
                return observations, reward, terminated, truncated, info

            for transbot_id in range(self.n_transbots):
                self.transbot_features[transbot_id, 6] = max(-1.0,
                                                             (self.empty_moving_time_matrix[
                                                                 int(self.transbot_features[transbot_id, 4]),
                                                                 int(self.job_features[self.chosen_job, 3])]) /
                                                             self.transbot_features[transbot_id, 1])

            self.stage = 2

        else:
            self.chosen_transbot = action["agent2"]

            tspt_result = self._schedule_tspt_task(
                job_id=self.chosen_job,
                task_id=self.tspt_task_id,
                transbot_id=self.chosen_transbot,
                machine_id=self.chosen_machine
            )

            prcs_result = self._schedule_prcs_task(
                job_id=self.chosen_job,
                task_id=self.prcs_task_id,
                machine_id=self.chosen_machine
            )

            # self.perform_left_shift_if_possible = True
            self.env_current_time = max(self.machine_features[:self.n_machines, 3])
            for job_id in range(self.n_jobs):
                if (self.job_arrival_time[job_id] < self.env_current_time) and self.job_features[job_id, 1] == 0:
                    self.job_action_mask[job_id] = 1

            self.machine_features[:self.n_machines, 5:] = -1
            self.transbot_features[:self.n_transbots, 6] = -1

            self.prev_cmax = self.curr_cmax
            self.curr_cmax = self.result_finish_time_for_jobs.max()

            self.reward_this_step = self.reward_this_step + 1.0 * (
                        self.prev_cmax - self.curr_cmax) / self.rule_makespan_for_current_instance
            # self.reward_this_step = self.reward_this_step + (self.prev_cmax - self.curr_cmax)

            self.schedule_done = min(self.job_features[:, 4]) == 1
            if self.schedule_done:
                self.final_makespan = self.curr_cmax
                self.drl_minus_rule = self.final_makespan - self.rule_makespan_for_current_instance
                self.reward_this_step = self.reward_this_step + 1

            for i in range(3):
                reward["agent{}".format(i)] = self.reward_this_step
                terminated["agent{}".format(i)] = self.schedule_done
                if terminated["agent{}".format(i)]:
                    self.terminateds.add("agent{}".format(i))
                truncated["agent{}".format(i)] = False

            self.stage = 0

        observations = self._get_obs()
        info = self._get_info()
        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = False

        return observations, reward, terminated, truncated, info

    # *********************************
    # Schedule An Transportation Task
    # *********************************
    def _schedule_tspt_task(self, job_id: int, task_id: int, transbot_id: int, machine_id: int) -> dict:
        """
        schedules a transport task/node in the graph representation if the task can be scheduled.

        This adding one or multiple corresponding edges (multiple when performing a left shift) and updating the
        information stored in the nodes.

        :param task_id:     the transport task or node that shall be scheduled.
        :return:            a dict with additional information for the `DisjunctiveGraphJssEnv.step`-method.
        """

        if self.Graph.nodes[task_id]["node_type"] != "operation" or \
                job_id != self.Graph.nodes[task_id]["job_id"] or \
                self.Graph.nodes[task_id]["operation_id"] != self.job_features[job_id][1]:
            return {
                "schedule_success": False,
            }

        if machine_id < 0 or machine_id >= self.n_machines or transbot_id < 0 or transbot_id >= self.n_transbots:
            return {
                "schedule_success": False,
            }

        transbot_location = int(self.transbot_features[transbot_id, 4])
        job_location = int(self.job_features[job_id, 3])
        machine_location = int(self.machine_features[machine_id, 0])

        previous_operation_finish_time = float(self.job_features[job_id, 2])

        expected_duration = self.empty_moving_time_matrix[transbot_location, job_location] + \
                            self.transport_time_matrix[job_location, machine_location]
        if self.transbot_quality[0, transbot_id] > 0:
            if random.random() <= self.transbot_quality[0, transbot_id]:
                actual_duration = expected_duration
            else:
                actual_duration = random.uniform(expected_duration,
                                                 expected_duration / self.transbot_quality[0, transbot_id])
        else:
            actual_duration = 999

        len_transbot_routes = len(self.transbot_routes[transbot_id])
        if len_transbot_routes:
            if self.perform_left_shift_if_possible:
                j_lower_bound_st = previous_operation_finish_time
                j_lower_bound_ft = j_lower_bound_st + actual_duration

                # check if task can be scheduled between src and first task
                transbot_firt_task = self.transbot_routes[transbot_id][0]
                transbot_first_task_start_time = self.result_start_time_for_jobs[
                    self.task_job_operation_dict[transbot_firt_task][0],
                    self.task_job_operation_dict[transbot_firt_task][1], 0]

                if j_lower_bound_ft <= transbot_first_task_start_time:
                    self.transbot_routes[transbot_id] = np.insert(self.transbot_routes[transbot_id], 0, task_id)
                    start_time = previous_operation_finish_time
                    finish_time = start_time + actual_duration

                    self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 0] = start_time
                    self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 0] = finish_time
                    self.Graph.nodes[task_id]["t_start_time"] = start_time
                    self.Graph.nodes[task_id]["t_finish_time"] = finish_time
                    self.Graph.nodes[task_id]["transbot_id"] = transbot_id
                    self.transbot_features[transbot_id][2] += 1
                    self.transbot_features[transbot_id][3] = finish_time
                    self.transbot_features[transbot_id][4] = machine_location
                    self.transbot_features[transbot_id][5] += actual_duration

                    self.job_features[job_id][2] = finish_time
                    self.job_features[job_id][3] = machine_location

                    return {
                        "schedule_success": True,
                        "actual_duration": actual_duration,
                        "start_time": start_time,
                        "finish_time": finish_time,
                    }

                elif len_transbot_routes == 1:
                    self.transbot_routes[transbot_id] = np.append(self.transbot_routes[transbot_id], task_id)
                    transbot_previous_task_finish_time = float(self.transbot_features[transbot_id, 3])
                    start_time = max(previous_operation_finish_time, transbot_previous_task_finish_time)
                    finish_time = start_time + actual_duration

                    self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 0] = start_time
                    self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 0] = finish_time
                    self.Graph.nodes[task_id]["t_start_time"] = start_time
                    self.Graph.nodes[task_id]["t_finish_time"] = finish_time
                    self.Graph.nodes[task_id]["transbot_id"] = transbot_id
                    self.transbot_features[transbot_id][2] += 1
                    self.transbot_features[transbot_id][3] = finish_time
                    self.transbot_features[transbot_id][4] = machine_location
                    self.transbot_features[transbot_id][5] += actual_duration

                    self.job_features[job_id][2] = finish_time
                    self.job_features[job_id][3] = machine_location

                    return {
                        "schedule_success": True,
                        "actual_duration": actual_duration,
                        "start_time": start_time,
                        "finish_time": finish_time,
                    }

                # check if task can be scheduled between two tasks
                for i, (m_prev, m_next) in enumerate(
                        zip(self.transbot_routes[transbot_id], self.transbot_routes[transbot_id][1:])):
                    m_temp_prev_ft = self.result_finish_time_for_jobs[
                        self.task_job_operation_dict[m_prev][0],
                        self.task_job_operation_dict[m_prev][1], 0]
                    m_temp_next_st = self.result_start_time_for_jobs[
                        self.task_job_operation_dict[m_next][0],
                        self.task_job_operation_dict[m_next][1], 0]

                    if j_lower_bound_ft > m_temp_next_st:
                        continue

                    m_gap = m_temp_next_st - m_temp_prev_ft
                    if m_gap < actual_duration:
                        continue

                    # at this point the task can fit in between two already scheduled task
                    start_time = max(j_lower_bound_st, m_temp_prev_ft)
                    finish_time = start_time + actual_duration
                    # insert task at the corresponding place in the machine routes list
                    self.transbot_routes[transbot_id] = np.insert(self.transbot_routes[transbot_id], i + 1, task_id)

                    self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 0] = start_time
                    self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 0] = finish_time
                    self.Graph.nodes[task_id]["t_start_time"] = start_time
                    self.Graph.nodes[task_id]["t_finish_time"] = finish_time
                    self.Graph.nodes[task_id]["transbot_id"] = transbot_id
                    self.transbot_features[transbot_id][2] += 1
                    self.transbot_features[transbot_id][3] = finish_time
                    self.transbot_features[transbot_id][4] = machine_location
                    self.transbot_features[transbot_id][5] += actual_duration

                    self.job_features[job_id][2] = finish_time
                    self.job_features[job_id][3] = machine_location

                    return {
                        "schedule_success": True,
                        "actual_duration": actual_duration,
                        "start_time": start_time,
                        "finish_time": finish_time,
                    }

                self.transbot_routes[transbot_id] = np.append(self.transbot_routes[transbot_id], task_id)
                transbot_previous_task_finish_time = float(self.transbot_features[transbot_id, 3])
                start_time = max(previous_operation_finish_time, transbot_previous_task_finish_time)
                finish_time = start_time + actual_duration

            else:
                self.transbot_routes[transbot_id] = np.append(self.transbot_routes[transbot_id], task_id)
                transbot_previous_task_finish_time = float(self.transbot_features[transbot_id, 3])
                start_time = max(previous_operation_finish_time, transbot_previous_task_finish_time)
                finish_time = start_time + actual_duration
        else:
            self.transbot_routes[transbot_id] = np.insert(self.transbot_routes[transbot_id], 0, task_id)
            start_time = previous_operation_finish_time
            finish_time = start_time + actual_duration

        self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 0] = start_time
        self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 0] = finish_time
        self.Graph.nodes[task_id]["t_start_time"] = start_time
        self.Graph.nodes[task_id]["t_finish_time"] = finish_time
        self.Graph.nodes[task_id]["transbot_id"] = transbot_id
        self.transbot_features[transbot_id][2] += 1
        self.transbot_features[transbot_id][3] = finish_time
        self.transbot_features[transbot_id][4] = machine_location
        self.transbot_features[transbot_id][5] += actual_duration

        self.job_features[job_id][2] = finish_time
        self.job_features[job_id][3] = machine_location

        return {
            "schedule_success": True,
            "actual_duration": actual_duration,
            "start_time": start_time,
            "finish_time": finish_time,
        }

    # *********************************
    # Schedule An Processing Task
    # *********************************
    def _schedule_prcs_task(self, job_id: int, task_id: int, machine_id: int) -> dict:
        """
        schedules a process task/node in the graph representation if the task can be scheduled.

        This adding one or multiple corresponding edges (multiple when performing a left shift) and updating the
        information stored in the nodes.

        :param task_id:     the process task or node that shall be scheduled.
        :param machine_id:     the process machine that shall be allocated.
        :return:            a dict with additional information for the `DisjunctiveGraphJssEnv.step`-method.
        """

        if self.Graph.nodes[task_id]["node_type"] != "operation" or \
                job_id != self.Graph.nodes[task_id]["job_id"] or \
                self.Graph.nodes[task_id]["operation_id"] != self.job_features[job_id][1]:
            return {
                "schedule_success": False,
            }

        if machine_id < 0 or machine_id >= self.n_machines:
            return {
                "schedule_success": False,
            }

        # previous_operation_finish_time = self.observations["job_attrs"][job_id][2] * self.min_max_total_time_for_a_job
        previous_operation_finish_time = float(self.job_features[job_id][2])

        expected_duration = self.processing_time_matrix[job_id, self.Graph.nodes[task_id]["operation_id"], machine_id]
        if expected_duration <= 0:
            return {
                "schedule_success": False,
            }
        if self.machine_quality[0, machine_id] > 0:
            if random.random() <= self.machine_quality[0, machine_id]:
                actual_duration = expected_duration
            else:
                actual_duration = random.uniform(expected_duration,
                                                 expected_duration / self.machine_quality[0, machine_id])
        else:
            actual_duration = 999

        len_machine_routes = len(self.machine_routes[machine_id])
        if len_machine_routes:
            if self.perform_left_shift_if_possible:
                j_lower_bound_st = previous_operation_finish_time
                j_lower_bound_ft = j_lower_bound_st + actual_duration

                # check if task can be scheduled between src and first task
                machine_firt_task = self.machine_routes[machine_id][0]
                machine_first_task_start_time = self.result_start_time_for_jobs[
                    self.task_job_operation_dict[machine_firt_task][0],
                    self.task_job_operation_dict[machine_firt_task][1], 1]

                if j_lower_bound_ft <= machine_first_task_start_time:
                    self.machine_routes[machine_id] = np.insert(self.machine_routes[machine_id], 0, task_id)
                    start_time = previous_operation_finish_time
                    finish_time = start_time + actual_duration

                    self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = start_time
                    self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = finish_time
                    self.Graph.nodes[task_id]["m_start_time"] = start_time
                    self.Graph.nodes[task_id]["m_finish_time"] = finish_time
                    self.Graph.nodes[task_id]["machine_id"] = machine_id
                    self.machine_features[machine_id][2] += 1
                    self.machine_features[machine_id][3] = finish_time
                    self.machine_features[machine_id][4] += actual_duration

                    self.job_features[job_id][1] += 1
                    self.job_features[job_id][2] = finish_time
                    self.job_features[job_id][5] += actual_duration
                    self.job_features[job_id][7] -= self.job_features[job_id, 6]
                    if self.job_features[job_id][7] < 0:
                        self.job_features[job_id][7] = 0
                    if self.job_features[job_id][1] >= self.n_operations_for_jobs[job_id]:
                        self.job_features[job_id, 4] = 1
                        self.job_features[job_id, 6] = -1
                        self.job_action_mask[job_id] = 0
                    else:
                        self.job_features[job_id, 6] = self.mean_processing_time_of_operations[job_id][
                            int(self.job_features[job_id][1])]

                    return {
                        "schedule_success": True,
                        "actual_duration": actual_duration,
                        "start_time": start_time,
                        "finish_time": finish_time,
                    }

                elif len_machine_routes == 1:
                    self.machine_routes[machine_id] = np.append(self.machine_routes[machine_id], task_id)
                    machine_previous_task_finish_time = float(self.machine_features[machine_id, 3])
                    start_time = max(previous_operation_finish_time, machine_previous_task_finish_time)
                    finish_time = start_time + actual_duration

                    self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = start_time
                    self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = finish_time
                    self.Graph.nodes[task_id]["start_time"] = start_time
                    self.Graph.nodes[task_id]["finish_time"] = finish_time
                    self.Graph.nodes[task_id]["machine_id"] = machine_id
                    self.machine_features[machine_id][2] += 1
                    self.machine_features[machine_id][3] = finish_time
                    self.machine_features[machine_id][4] += actual_duration

                    self.job_features[job_id][1] += 1
                    self.job_features[job_id][2] = finish_time
                    self.job_features[job_id][5] += actual_duration
                    self.job_features[job_id][7] -= self.job_features[job_id, 6]
                    if self.job_features[job_id][7] < 0:
                        self.job_features[job_id][7] = 0
                    if self.job_features[job_id][1] >= self.n_operations_for_jobs[job_id]:
                        self.job_features[job_id, 4] = 1
                        self.job_features[job_id, 6] = -1
                        self.job_action_mask[job_id] = 0
                    else:
                        self.job_features[job_id, 6] = self.mean_processing_time_of_operations[job_id][
                            int(self.job_features[job_id][1])]

                    return {
                        "schedule_success": True,
                        "actual_duration": actual_duration,
                        "start_time": start_time,
                        "finish_time": finish_time,
                    }

                # check if task can be scheduled between two tasks
                for i, (m_prev, m_next) in enumerate(
                        zip(self.machine_routes[machine_id], self.machine_routes[machine_id][1:])):
                    m_temp_prev_ft = self.result_finish_time_for_jobs[
                        self.task_job_operation_dict[m_prev][0],
                        self.task_job_operation_dict[m_prev][1], 1]
                    m_temp_next_st = self.result_start_time_for_jobs[
                        self.task_job_operation_dict[m_next][0],
                        self.task_job_operation_dict[m_next][1], 1]

                    if j_lower_bound_ft > m_temp_next_st:
                        continue

                    m_gap = m_temp_next_st - m_temp_prev_ft
                    if m_gap < actual_duration:
                        continue

                    # at this point the task can fit in between two already scheduled task
                    start_time = max(j_lower_bound_st, m_temp_prev_ft)
                    finish_time = start_time + actual_duration
                    # insert task at the corresponding place in the machine routes list
                    self.machine_routes[machine_id] = np.insert(self.machine_routes[machine_id], i + 1, task_id)

                    self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = start_time
                    self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = finish_time
                    self.Graph.nodes[task_id]["m_start_time"] = start_time
                    self.Graph.nodes[task_id]["m_finish_time"] = finish_time
                    self.Graph.nodes[task_id]["machine_id"] = machine_id
                    self.machine_features[machine_id][2] += 1
                    self.machine_features[machine_id][3] = finish_time
                    self.machine_features[machine_id][4] += actual_duration

                    self.job_features[job_id][1] += 1
                    self.job_features[job_id][2] = finish_time
                    self.job_features[job_id][5] += actual_duration
                    self.job_features[job_id][7] -= self.job_features[job_id, 6]
                    if self.job_features[job_id][7] < 0:
                        self.job_features[job_id][7] = 0
                    if self.job_features[job_id][1] >= self.n_operations_for_jobs[job_id]:
                        self.job_features[job_id, 4] = 1
                        self.job_features[job_id, 6] = -1
                        self.job_action_mask[job_id] = 0
                    else:
                        self.job_features[job_id, 6] = self.mean_processing_time_of_operations[job_id][
                            int(self.job_features[job_id][1])]

                    return {
                        "schedule_success": True,
                        "actual_duration": actual_duration,
                        "start_time": start_time,
                        "finish_time": finish_time,
                    }

                self.machine_routes[machine_id] = np.append(self.machine_routes[machine_id], task_id)
                machine_previous_task_finish_time = float(self.machine_features[machine_id][3])
                start_time = max(previous_operation_finish_time, machine_previous_task_finish_time)
                finish_time = start_time + actual_duration

            else:
                self.machine_routes[machine_id] = np.append(self.machine_routes[machine_id], task_id)
                machine_previous_task_finish_time = float(self.machine_features[machine_id][3])
                start_time = max(previous_operation_finish_time, machine_previous_task_finish_time)
                finish_time = start_time + actual_duration
        else:
            self.machine_routes[machine_id] = np.insert(self.machine_routes[machine_id], 0, task_id)
            start_time = previous_operation_finish_time
            finish_time = start_time + actual_duration

        self.result_start_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = start_time
        self.result_finish_time_for_jobs[job_id, self.Graph.nodes[task_id]["operation_id"], 1] = finish_time
        self.Graph.nodes[task_id]["m_start_time"] = start_time
        self.Graph.nodes[task_id]["m_finish_time"] = finish_time
        self.Graph.nodes[task_id]["machine_id"] = machine_id
        self.machine_features[machine_id][2] += 1
        self.machine_features[machine_id][3] = finish_time
        self.machine_features[machine_id][4] += actual_duration

        self.job_features[job_id][1] += 1
        self.job_features[job_id][2] = finish_time
        self.job_features[job_id][5] += actual_duration
        self.job_features[job_id][7] -= self.job_features[job_id, 6]
        if self.job_features[job_id][7] < 0:
            self.job_features[job_id][7] = 0
        if self.job_features[job_id][1] >= self.n_operations_for_jobs[job_id]:
            self.job_features[job_id, 4] = 1
            self.job_features[job_id, 6] = -1
            self.job_action_mask[job_id] = 0
        else:
            self.job_features[job_id, 6] = self.mean_processing_time_of_operations[job_id][
                int(self.job_features[job_id][1])]

        return {
            "schedule_success": True,
            "actual_duration": actual_duration,
            "start_time": start_time,
            "finish_time": finish_time,
        }

    def initialize_disjunctive_graph(self,
                                     n_operations_for_jobs,
                                     # process_time_matrix,
                                     ) -> None:
        """
        Get a new disjunctive graph.
        """
        self.Graph = nx.DiGraph()

        # add nodes for processing machines
        for machine_id in range(self.n_machines):
            self.Graph.add_node(
                "machine" + str(machine_id),
                machine_id=machine_id,
                node_type="machine",
                shape='p',
                pos=(machine_id, -self.n_jobs - 2),
            )

        # add nodes for transbots
        for transbot_id in range(self.n_transbots):
            self.Graph.add_node(
                "transbot" + str(transbot_id),
                transbot_id=transbot_id,
                node_type="transbot",
                shape='p',
                pos=(transbot_id, 0),
            )

        # add src node
        self.Graph.add_node(
            self.source_task,
            node_type="dummy",
            shape='o',
            pos=(-2, int(-self.n_jobs * 0.5) - 1),
        )
        # add sink task at the end to avoid permutation in the adj matrix.
        self.Graph.add_node(
            self.sink_task,
            node_type="dummy",
            shape='o',
            pos=(2 * max(n_operations_for_jobs), int(-self.n_jobs * 0.5) - 1),
        )

        # add nodes for tasks
        task_id = -1
        self.task_job_operation_dict = {}
        for job_id in range(self.n_jobs):
            for operation_id in range(n_operations_for_jobs[job_id]):
                task_id += 1  # start from transportation task id 0, -1 is dummy starting task
                self.task_job_operation_dict[task_id] = [job_id, operation_id]
                # add a transportation task node
                self.Graph.add_node(
                    task_id,
                    node_type="operation",
                    job_id=job_id,
                    operation_id=operation_id,
                    shape='s',
                    pos=(2 * operation_id - 1, -job_id - 1),
                    # start_time=-1,
                    # finish_time=-1,
                    t_start_time=-1,
                    t_finish_time=-1,
                    transbot_id=-1,
                    m_start_time=-1,
                    m_finish_time=-1,
                    machine_id=-1,
                )
                if operation_id != 0:
                    # add a conjunctive edge from last processing (task_id - 1) to this transportation (task_id)
                    self.Graph.add_edge(
                        task_id - 1, task_id,
                        edge_type="conjunctive_arc",
                        weight=0,
                    )

