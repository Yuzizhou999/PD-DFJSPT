import numpy as np


def _machine_EET_scores(legal_machine_actions, real_machine_attrs, pref_vec=None):
    """Composite EET scores with invalid entries masked."""

    pref_vec = np.asarray(pref_vec, dtype=np.float32) if pref_vec is not None else np.array([1.0, 1.0], dtype=np.float32)
    machine_actions_mask = (1 - legal_machine_actions) * 1e8
    # Estimated completion combines machine availability, processing time, and transport.
    expected_completion = real_machine_attrs[:, 3] + real_machine_attrs[:, 5] + real_machine_attrs[:, 6]
    waiting_penalty = real_machine_attrs[:, 3]
    congestion_cost = real_machine_attrs[:, 2]

    composite_cost = pref_vec[0] * expected_completion + pref_vec[1] * (waiting_penalty + congestion_cost)
    composite_cost += machine_actions_mask

    scores = np.full_like(composite_cost, -np.inf, dtype=np.float32)
    valid_indices = np.where(legal_machine_actions > 0)[0]
    scores[valid_indices] = -composite_cost[valid_indices]
    return scores


def machine_EET_action(legal_machine_actions, real_machine_attrs, pref_vec=None):
    scores = _machine_EET_scores(legal_machine_actions, real_machine_attrs, pref_vec)
    best_score = np.max(scores)
    best_indices = np.where(scores == best_score)[0]
    EET_machine_action = {
        "agent1": int(np.random.choice(best_indices))
    }
    return EET_machine_action


def machine_SPT_action(legal_machine_actions, real_machine_attrs):
    machine_actions_mask = (1 - legal_machine_actions) * 1e8
    machine_processing_time = real_machine_attrs[:, 5]
    # machine_processing_time = np.zeros(len(legal_machine_actions))
    # machine_processing_time[:len(real_machine_attrs)] = [obs[5] for obs in real_machine_attrs]
    machine_processing_time += machine_actions_mask
    SPT_machine_action = {
        "agent1": np.argmin(machine_processing_time)
    }
    return SPT_machine_action


def _transbot_EET_scores(legal_transbot_actions, real_transbot_attrs, pref_vec=None):
    """Composite transbot scores with invalid entries masked."""

    pref_vec = np.asarray(pref_vec, dtype=np.float32) if pref_vec is not None else np.array([1.0, 1.0], dtype=np.float32)
    transbot_actions_mask = (1 - legal_transbot_actions) * 1e8
    expected_completion = real_transbot_attrs[:, 3] + real_transbot_attrs[:, 6]
    waiting_penalty = real_transbot_attrs[:, 3]
    congestion_cost = real_transbot_attrs[:, 2]

    composite_cost = pref_vec[0] * expected_completion + pref_vec[1] * (waiting_penalty + congestion_cost)
    composite_cost += transbot_actions_mask

    scores = np.full_like(composite_cost, -np.inf, dtype=np.float32)
    valid_indices = np.where(legal_transbot_actions > 0)[0]
    scores[valid_indices] = -composite_cost[valid_indices]
    return scores


def transbot_EET_action(legal_transbot_actions, real_transbot_attrs, pref_vec=None):
    scores = _transbot_EET_scores(legal_transbot_actions, real_transbot_attrs, pref_vec)
    best_score = np.max(scores)
    best_indices = np.where(scores == best_score)[0]
    EET_transbot_action = {
        "agent2": int(np.random.choice(best_indices))
    }
    return EET_transbot_action


def transbot_SPT_action(legal_transbot_actions, real_transbot_attrs):
    transbot_actions_mask = (1 - legal_transbot_actions) * 1e8
    transbot_transporting_time = real_transbot_attrs[:, 6]
    # transbot_transporting_time = np.zeros(len(legal_transbot_actions))
    # transbot_transporting_time[:len(real_transbot_attrs)] = [obs[6] for obs in real_transbot_attrs]
    transbot_transporting_time += transbot_actions_mask
    SPT_transbot_action = {
        "agent2": np.argmin(transbot_transporting_time)
    }
    return SPT_transbot_action