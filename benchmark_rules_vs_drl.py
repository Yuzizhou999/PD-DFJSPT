"""Benchmark DRL policies against heuristic dispatching rules.

This script evaluates a trained DRL checkpoint and the built-in heuristic
rules on the same test instances so their makespan and tardiness can be
compared side-by-side.
"""

import argparse
import copy
import os
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import ray
from ray.rllib import Policy
from ray.rllib.models import ModelCatalog

from DFJSPT import dfjspt_params
from DFJSPT.dfjspt_agent_model import (
    JobActionMaskModel,
    MachineActionMaskModel,
    TransbotActionMaskModel,
)
from DFJSPT.dfjspt_env import DfjsptMaEnv
from DFJSPT.dfjspt_rule.job_selection_rules import (
    job_EST_action,
    job_MTWR_action,
    job_SPT_action,
)
from DFJSPT.dfjspt_rule.machine_selection_rules import (
    machine_EET_action,
    machine_SPT_action,
    transbot_EET_action,
    transbot_SPT_action,
)

JobSelector = Callable[..., Dict[str, int]]
MachineSelector = Callable[[np.ndarray, np.ndarray], Dict[str, int]]
TransbotSelector = Callable[[np.ndarray, np.ndarray], Dict[str, int]]


def mopnr_job_action(
    legal_job_actions: np.ndarray,
    real_job_attrs: np.ndarray,
    n_jobs: int,
    n_operations_for_jobs: np.ndarray,
) -> Dict[str, int]:
    """Most Operations Remaining (MOPNR) job selector used by rules 3 and 4."""
    job_actions_mask = (1 - legal_job_actions) * 1e8
    jobs_progress = -job_actions_mask
    jobs_progress[:n_jobs] += n_operations_for_jobs - real_job_attrs[:n_jobs, 1]
    return {"agent0": int(np.argmax(jobs_progress))}


RULE_DEFINITIONS: Dict[int, Tuple[JobSelector, MachineSelector, TransbotSelector]] = {
    1: (job_EST_action, machine_EET_action, transbot_EET_action),
    2: (job_EST_action, machine_SPT_action, transbot_SPT_action),
    3: (
        lambda legal, attrs, env: mopnr_job_action(
            legal, attrs, env.n_jobs, env.n_operations_for_jobs
        ),
        machine_EET_action,
        transbot_EET_action,
    ),
    4: (
        lambda legal, attrs, env: mopnr_job_action(
            legal, attrs, env.n_jobs, env.n_operations_for_jobs
        ),
        machine_SPT_action,
        transbot_SPT_action,
    ),
    5: (job_SPT_action, machine_EET_action, transbot_EET_action),
    6: (job_SPT_action, machine_SPT_action, transbot_SPT_action),
    7: (job_MTWR_action, machine_EET_action, transbot_EET_action),
    8: (job_MTWR_action, machine_SPT_action, transbot_SPT_action),
}


def _step_heuristic(
    env: DfjsptMaEnv,
    selectors: Tuple[JobSelector, MachineSelector, TransbotSelector],
    observation: Dict,
    info: Dict,
) -> None:
    stage = next(iter(info["agent0"].values()), None)
    done = False

    while not done:
        if stage == 0:
            legal_job_actions = copy.deepcopy(observation["agent0"]["action_mask"])
            real_job_attrs = copy.deepcopy(observation["agent0"]["observation"])
            try:
                action = selectors[0](legal_job_actions, real_job_attrs, env)
            except TypeError:
                # Backward compatibility for job rules that accept only two arguments
                action = selectors[0](legal_job_actions, real_job_attrs)
            observation, reward, terminated, truncated, info = env.step(action)
            stage = next(iter(info["agent1"].values()), None)
        elif stage == 1:
            legal_machine_actions = copy.deepcopy(observation["agent1"]["action_mask"])
            real_machine_attrs = copy.deepcopy(observation["agent1"]["observation"])
            action = selectors[1](legal_machine_actions, real_machine_attrs)
            observation, reward, terminated, truncated, info = env.step(action)
            stage = next(iter(info["agent2"].values()), None)
        else:
            legal_transbot_actions = copy.deepcopy(observation["agent2"]["action_mask"])
            real_transbot_attrs = copy.deepcopy(observation["agent2"]["observation"])
            action = selectors[2](legal_transbot_actions, real_transbot_attrs)
            observation, reward, terminated, truncated, info = env.step(action)
            stage = next(iter(info["agent0"].values()), None)
            done = terminated["__all__"]


def run_heuristic(rule_id: int, env: DfjsptMaEnv, instance_id: int) -> Tuple[float, float]:
    if rule_id not in RULE_DEFINITIONS:
        raise ValueError(f"Unsupported rule id: {rule_id}")

    observation, info = env.reset(options={"instance_id": instance_id})
    selectors = RULE_DEFINITIONS[rule_id]
    _step_heuristic(env, selectors, observation, info)
    return float(env.final_makespan), float(env.curr_tardiness)


def run_drl(
    policies: Tuple[Policy, Policy, Policy],
    env: DfjsptMaEnv,
    instance_id: int,
) -> Tuple[float, float]:
    observation, info = env.reset(options={"instance_id": instance_id})
    stage = next(iter(info["agent0"].values()), None)
    done = False

    while not done:
        if stage == 0:
            action = {"agent0": policies[0].compute_single_action(obs=observation["agent0"], explore=False)[0]}
            observation, reward, terminated, truncated, info = env.step(action)
            stage = next(iter(info["agent1"].values()), None)
        elif stage == 1:
            action = {"agent1": policies[1].compute_single_action(obs=observation["agent1"], explore=False)[0]}
            observation, reward, terminated, truncated, info = env.step(action)
            stage = next(iter(info["agent2"].values()), None)
        else:
            action = {"agent2": policies[2].compute_single_action(obs=observation["agent2"], explore=False)[0]}
            observation, reward, terminated, truncated, info = env.step(action)
            stage = next(iter(info["agent0"].values()), None)
            done = terminated["__all__"]

    return float(env.final_makespan), float(env.curr_tardiness)


def evaluate_rules(rule_ids: Iterable[int], instances: int, runs_per_instance: int) -> pd.DataFrame:
    env = DfjsptMaEnv({"train_or_eval_or_test": "test"})
    records: List[Dict] = []

    for rule_id in rule_ids:
        for instance_id in range(instances):
            for repeat in range(runs_per_instance):
                makespan, tardiness = run_heuristic(rule_id, env, instance_id)
                records.append(
                    {
                        "method": f"rule_{rule_id}",
                        "rule_id": rule_id,
                        "instance_id": instance_id,
                        "run": repeat,
                        "makespan": makespan,
                        "total_tardiness": tardiness,
                    }
                )

    return pd.DataFrame.from_records(records)


def evaluate_drl(checkpoint: str, instances: int, runs_per_instance: int) -> pd.DataFrame:
    ray.init(local_mode=False, include_dashboard=False)
    ModelCatalog.register_custom_model("job_agent_model", JobActionMaskModel)
    ModelCatalog.register_custom_model("machine_agent_model", MachineActionMaskModel)
    ModelCatalog.register_custom_model("transbot_agent_model", TransbotActionMaskModel)

    job_policy = Policy.from_checkpoint(os.path.join(checkpoint, "policies", "policy_agent0"))
    machine_policy = Policy.from_checkpoint(os.path.join(checkpoint, "policies", "policy_agent1"))
    transbot_policy = Policy.from_checkpoint(os.path.join(checkpoint, "policies", "policy_agent2"))

    env = DfjsptMaEnv({"train_or_eval_or_test": "test"})
    records: List[Dict] = []

    for instance_id in range(instances):
        for repeat in range(runs_per_instance):
            makespan, tardiness = run_drl(
                (job_policy, machine_policy, transbot_policy), env, instance_id
            )
            records.append(
                {
                    "method": "drl",
                    "instance_id": instance_id,
                    "run": repeat,
                    "makespan": makespan,
                    "total_tardiness": tardiness,
                }
            )

    ray.shutdown()
    return pd.DataFrame.from_records(records)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["method"])
        .agg(
            avg_makespan=("makespan", "mean"),
            std_makespan=("makespan", "std"),
            avg_tardiness=("total_tardiness", "mean"),
            std_tardiness=("total_tardiness", "std"),
            samples=("makespan", "count"),
        )
        .reset_index()
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare heuristic rules against a DRL checkpoint on DFJSPT.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the DRL checkpoint directory containing policy_agent* folders.",
    )
    parser.add_argument(
        "--rules",
        type=int,
        nargs="*",
        default=list(RULE_DEFINITIONS.keys()),
        help="Rule IDs to evaluate (default: all).",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=dfjspt_params.n_instances_for_testing,
        help="Number of test instances to evaluate.",
    )
    parser.add_argument(
        "--drl-runs",
        type=int,
        default=1,
        help="Number of repeated runs per instance for the DRL policy (set >1 to match rule repetitions).",
    )
    parser.add_argument(
        "--rule-runs",
        type=int,
        default=5,
        help="Number of repeated runs per instance for each rule (to smooth randomness).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "benchmark_results", "heuristic_vs_drl.csv"),
        help="Where to store the detailed result CSV.",
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    drl_df = evaluate_drl(args.checkpoint, args.instances, args.drl_runs)
    rules_df = evaluate_rules(args.rules, args.instances, args.rule_runs)
    results = pd.concat([drl_df, rules_df], ignore_index=True)
    summary = summarize_results(results)

    results.to_csv(args.output, index=False)
    summary_path = os.path.splitext(args.output)[0] + "_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Detailed results saved to:", args.output)
    print("Summary saved to:", summary_path)
    print("\nSummary statistics:\n", summary)


if __name__ == "__main__":
    main()