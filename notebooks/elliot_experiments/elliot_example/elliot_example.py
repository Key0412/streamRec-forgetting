import os
import json
import yaml
import sys

from elliot.run import run_experiment

import tensorflow as tf
from tensorflow.python.client import device_lib
tf.autograph.set_verbosity(5)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()

def getBestModelParams(path_to_results):
    for item in os.listdir(path_to_results + 'performance/'):
        if 'best' in item:
            with open(path_to_results + 'performance/' + item) as file:
                best_model_info = json.load(file)
            break

    best_model_params = best_model_info[1]['configuration'] # type: ignore
    model = best_model_params['name'].split('_')[0]
    del best_model_params['name'], best_model_params['best_iteration']
    return model, best_model_params

def setNewConfig( path_to_config_file, model_tup=None, path_to_results=None):
    assert model_tup or path_to_results, 'At least one of model_tup or path_to_results must be passed'

    with open(path_to_config_file, "r") as stream:
        try:
            yaml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit()
     
    bucket_idx = int(path_to_config_file[path_to_config_file.find('_b')+2]) # index of the bucket
    holdout_idx = int(path_to_config_file[path_to_config_file.find('_h')+2]) # index of the last holdout tested
    
    if bucket_idx == 0:
        holdout_idx = str(holdout_idx + 1)
    elif holdout_idx == bucket_idx:
        holdout_idx = str(0)
    elif holdout_idx < bucket_idx:
        if (bucket_idx - holdout_idx) == 1:
            holdout_idx = str(holdout_idx + 2)
        else:
            holdout_idx = str(holdout_idx + 1)
    else:
        holdout_idx = str(holdout_idx + 1 )

    try:
        model, best_model_params = model_tup # type: ignore
    except:
        model, best_model_params = getBestModelParams(path_to_results)
        # new_path_to_results = path_to_results[:path_to_results.find('_h')+2] + holdout_idx + path_to_results[path_to_results.find('_h')+3:] # type: ignore
        # copy_tree(path_to_results, new_path_to_results)  # type: ignore

    # Update variables and files according to new holdout idx
    new_config_file = path_to_config_file[:path_to_config_file.find('_h')+2] + holdout_idx + path_to_config_file[path_to_config_file.find('_h')+3:]
    # for p in ['path_output_rec_result', 'path_output_rec_weight', 'path_output_rec_performance',  'path_log_folder']:
    #     p_path = yaml_file['experiment'][p]
    #     yaml_file['experiment'][ p ] = p_path[:p_path.find('_h')+2] + holdout_idx + p_path[p_path.find('_h')+3:]


    test_path = yaml_file['experiment']['data_config']['test_path']
    yaml_file['experiment']['data_config']['test_path'] = test_path[:test_path.find('_h')+2] + holdout_idx + test_path[test_path.find('_h')+3:]
    yaml_file['experiment']['models'][model].update(best_model_params)
    yaml_file['experiment']['models'][model]['meta'].update({'restore': True})       
    yaml_file['experiment']['models'][model]['meta'].update({'save_weights': False})   
    # yaml_file['experiment']['models'][model]['meta'].update({'save_recs': True}) # TEMP

    with open(new_config_file, "w") as stream:
        yaml.safe_dump(yaml_file, stream)

    return new_config_file, (model, best_model_params)

# Training/Evaluation 1
path_to_config_file = "/home/kpfra/streamRec-forgetting/notebooks/elliot_experiments/elliot_example/elliot_example_configuration_b0_h0.yml"
path_to_results = '/home/kpfra/streamRec-forgetting/results_b0_h0/'

run_experiment(path_to_config_file)
    
# Evaluation 2

path_to_config_file, model_tup = setNewConfig( path_to_config_file, path_to_results=path_to_results )
run_experiment(path_to_config_file)

# Evaluation 3

path_to_config_file, _ = setNewConfig( path_to_config_file, model_tup=model_tup )
run_experiment(path_to_config_file)