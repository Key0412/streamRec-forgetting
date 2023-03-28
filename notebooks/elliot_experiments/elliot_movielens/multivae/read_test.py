import os
import json
import yaml
import sys

path = '/home/kpfra/streamRec-forgetting/notebooks/elliot_experiments/elliot_movielens/multivae/sampled_movielens_bucket0_holdout0/'

for item in os.listdir(path + 'performance/'):
    if 'best' in item:
        with open(path + 'performance/' + item) as file:
            best_model_info = json.load(file)

best_model_params = best_model_info[1]['configuration'] # type: ignore
model = best_model_params['name'].split('_')[0]
del best_model_params['name']

with open("/home/kpfra/streamRec-forgetting/notebooks/elliot_experiments/elliot_movielens/multivae/ml_multivae_b0_h1_config.yml", "r") as stream:
    try:
        yaml_file = yaml.safe_load(stream)
        # print(yaml_file['experiment']['models'][model])
        yaml_file['experiment']['models'][model].update(best_model_params)
        # sms = '['
        # for i, m in enumerate( yaml_file['experiment']['evaluation']['simple_metrics'] ):
        #     if i == len(yaml_file['experiment']['evaluation']['simple_metrics']) - 1:
        #         sms += m + ']'    
        #     else:
        #         sms += m + ','
        # yaml_file['experiment']['evaluation']['simple_metrics'] = sms
        # print(yaml_file['experiment']['models'][model])
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit()

# print(yaml_file)

with open("/home/kpfra/streamRec-forgetting/notebooks/elliot_experiments/elliot_movielens/multivae/ml_multivae_b0_h1_config.yml", "w") as stream:
    try:
        yaml.safe_dump(yaml_file, stream)
    except yaml.YAMLError as exc:
        print(exc)