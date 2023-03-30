from sys import path
path.insert(0, '/home/kpfra/streamRec-forgetting/notebooks/elliot_experiments/')
from source import experiment

# Training/Evaluation 1
path_to_config_file = "/home/kpfra/streamRec-forgetting/notebooks/elliot_experiments/elliot_example/elliot_example_configuration_b0_h0.yml"
path_to_datasets = '/home/kpfra/streamRec-forgetting/notebooks/elliot_experiments/elliot_example'

experiment.run(path_to_config_file, path_to_datasets)