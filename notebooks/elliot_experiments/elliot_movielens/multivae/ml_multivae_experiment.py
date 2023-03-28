from elliot.run import run_experiment
from distutils.dir_util import copy_tree

import tensorflow as tf
from tensorflow.python.client import device_lib
tf.autograph.set_verbosity(5)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()

# Training/Evaluation: b0 - h0

run_experiment('ml_multivae_b0_h0_config.yml')

# Evaluation: b0 - h1
copy_tree("sampled_movielens_b0_h0", "sampled_movielens_b0_h1")
run_experiment('ml_multivae_b0_h1_config.yml')

# Evaluation: b0 - h2
copy_tree("sampled_movielens_b0_h0", "sampled_movielens_b0_h2")
run_experiment('ml_multivae_b0_h2_config.yml')

# print()
# print('START EXP 2')
# print()

# run_experiment('notebooks/elliot_experiments/elliot_example/elliot_example_configuration 2.yml')

