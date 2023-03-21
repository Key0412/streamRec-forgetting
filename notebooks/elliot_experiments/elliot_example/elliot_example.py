import tensorflow as tf
from tensorflow.python.client import device_lib
tf.autograph.set_verbosity(5)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()

import numpy as np 

def avg_recall(results_matrix): # Lopez-Paz e Ranzato GEM 2017
    return np.mean( np.diag(results_matrix) )

def compute_BWT(results_matrix): # Lopez-Paz e Ranzato GEM 2017
    BWT = []
    n_checkpoints = results_matrix.shape[0]
    for T in range(1, n_checkpoints): # 1 means holdout 2, 2 means 3, so on
        Rti = results_matrix.iloc[T, 0:T] # get models performances' on previous holdouts
        Rii = np.diag(results_matrix)[0:T] # get models performances' on their closest holdouts (diagonal)
        E = sum( Rti - Rii ) # future models performances' - performances' of models closest to holdouts (diagonal)
        BWT.append( E/T ) # store average BWT for model
    return BWT, np.mean( BWT ) # return BWT and average BWT for all models

def compute_FWT(results_matrix): # Díaz-Rodriguez et al. 2018
    upper_tri = results_matrix.to_numpy()[np.triu_indices(results_matrix.shape[0], k=1)]
    return np.mean(upper_tri)

from elliot.run import run_experiment

run_experiment('notebooks/elliot_experiments/elliot_example/elliot_example_configuration.yml')

# COPY RESULTS TO NEW FOLDER, WHICH IS TO BE READ FROM 
    # elliot_example_configuration 2 or Load Test
    
# from distutils.dir_util import copy_tree
# copy_tree("results", "results_cp")

# print()
# print('START EXP 2')
# print()

run_experiment('notebooks/elliot_experiments/elliot_example/elliot_example_configuration 2.yml')

# copy_tree("results", "results_cp2")
# run_experiment('elliot_example_configuration Load Test.yml')