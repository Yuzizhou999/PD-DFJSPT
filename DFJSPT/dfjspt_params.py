
# training params
as_test = False
framework = "torch"
local_mode = False
use_tune = True
use_custom_loss = True
use_her = True
il_loss_weight = 10.0
il_softmax_temperature_start = 2.0
il_softmax_temperature_end = 0.5
il_softmax_temperature_decay = 0.99
stop_iters = 6
stop_timesteps = 100000000000
stop_reward = 2
num_workers = 4
num_gpu = 0
num_envs_per_worker = 4


max_n_jobs = 10
n_jobs_is_fixed = True
n_jobs = 10
n_operations_is_n_machines = False
min_n_operations = 5
max_n_operations = 5
consider_job_insert = False
new_arrival_jobs = 3
earliest_arrive_time = 30
latest_arrive_time = 300

max_n_machines = 5
min_prcs_time = 1
max_prcs_time = 100
n_machines_is_fixed = True
n_machines = 5
is_fully_flexible = False
min_compatible_machines = 1
time_for_compatible_machines_are_same = False
time_viration_range = 5

max_n_transbots = 3
min_tspt_time = 1
max_tspt_time = 10
loaded_transport_time_scale = 1.5
n_transbots_is_fixed = True
n_transbots = 3

all_machines_are_perfect = False
min_quality = 0.1
# normalized_scale = max_n_operations * max_prcs_time

n_instances = 10200
n_instances_for_training = 5000
n_instances_for_evaluation = 100
n_instances_for_testing = 100
instance_generator_seed = 1000
layout_seed = 0

# env params
perform_left_shift_if_possible = True

# instance selection params
randomly_select_instance = True
current_instance_id = 0
imitation_env_count = 0
env_count = 0


# render params
JobAsAction = True
gantt_y_axis = "nJob"
drawMachineToPrcsEdges = True
default_visualisations = None




