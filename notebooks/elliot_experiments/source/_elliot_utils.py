import os
import json
import yaml
import sys

'''
Utilities to run elliot experiments over buckets and holdouts.
'''

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

def getBucketsNumber(path_to_datasets):
    files = os.listdir(path_to_datasets)
    n_buckets = max( [int(f[f.find('_')+2]) for f in files if f.endswith('.csv')] ) + 1
    return n_buckets

def setNewConfig( path_to_config_file, model_tup=None):
    # assert model_tup or path_to_results, 'At least one of model_tup or path_to_results must be passed'
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

    slice_pos = yaml_file['experiment']['path_output_rec_result'].rfind('/') + 1
    path_to_results = yaml_file['experiment']['path_output_rec_result'][:slice_pos]
    print('\n', f'Results will be stored in {path_to_results}'.center(100,'*'), end='\n')    
    
    try:
        model, best_model_params = model_tup # type: ignore
    except:
        model, best_model_params = getBestModelParams(path_to_results)
        

    # Update variables and files according to new holdout idx
    path_to_config_file = path_to_config_file[:path_to_config_file.find('_h')+2] + holdout_idx + path_to_config_file[path_to_config_file.find('_h')+3:]
    # update test path
    test_path = yaml_file['experiment']['data_config']['test_path']
    yaml_file['experiment']['data_config']['test_path'] = test_path[:test_path.find('_h')+2] + holdout_idx + test_path[test_path.find('_h')+3:]
    # update model params using previous best model
    yaml_file['experiment']['models'][model].update(best_model_params)
    yaml_file['experiment']['models'][model]['meta'].update({'restore': True})       
    yaml_file['experiment']['models'][model]['meta'].update({'save_weights': False})   
    # update dataset name
    dataset_name = yaml_file['experiment']['dataset']
    yaml_file['experiment']['dataset'] = dataset_name[:dataset_name.find('_h')+2] + holdout_idx
    # yaml_file['experiment']['models'][model]['meta'].update({'save_recs': True}) # TEMP

    with open(path_to_config_file, "w") as stream:
        yaml.safe_dump(yaml_file, stream)

    return path_to_config_file, (model, best_model_params)

def setNewBucketConfig(path_to_config_file, bucket_idx):
    with open(path_to_config_file, "r") as stream:
        try:
            yaml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit()
    # Update variables and files according to new training bucket and holdout
    bucket_idx = str(bucket_idx)
    paths_list = ['path_output_rec_result', 'path_output_rec_weight', 'path_output_rec_performance', 'path_log_folder']
    for tag in ['_h', '_b']:
        #update config file path
        path_to_config_file = path_to_config_file[:path_to_config_file.find(tag)+2] + bucket_idx + path_to_config_file[path_to_config_file.find(tag)+3:]
        # update output config:
        for p in paths_list:
            config_path = yaml_file['experiment'][p]
            yaml_file['experiment'][p] = config_path[:config_path.find(tag)+2] + bucket_idx + config_path[config_path.find(tag)+3:]
        # update dataset name
        dataset_name = yaml_file['experiment']['dataset']
        yaml_file['experiment']['dataset'] = dataset_name[:dataset_name.find(tag)+2] + bucket_idx + dataset_name[dataset_name.find(tag)+3:]
    # update train and test paths
    train_path = yaml_file['experiment']['data_config']['train_path']
    test_path = yaml_file['experiment']['data_config']['test_path']
    yaml_file['experiment']['data_config']['train_path'] = train_path[:train_path.find('_b')+2] + bucket_idx + train_path[train_path.find('_b')+3:]
    yaml_file['experiment']['data_config']['test_path'] = test_path[:test_path.find('_h')+2] + bucket_idx + test_path[test_path.find('_h')+3:]

    with open(path_to_config_file, "w") as stream: # type: ignore
        yaml.safe_dump(yaml_file, stream)

    return path_to_config_file # type: ignore

if __name__ == '__main__':
    path_to_config_file = "/home/kpfra/streamRec-forgetting/notebooks/elliot_experiments/elliot_example/elliot_example_configuration_b0_h0.yml"
    print(setNewBucketConfig(path_to_config_file, 1))
    print(setNewBucketConfig(path_to_config_file, 2))
    print(setNewBucketConfig(path_to_config_file, 10))
