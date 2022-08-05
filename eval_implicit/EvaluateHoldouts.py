from eval_implicit import EvalHoldout
from recommenders_implicit import *
import pandas as pd
import numpy as np
import copy
import time

class EvaluateHoldouts():
    '''
    Instanciation:\n
        \tIncremental training of recommendation model.\n
        \tStore model checkpoints at the end of each bucket.\n
    Methods:\n
        \tEvaluateHoldouts to evaluate models over holdouts - recall@N_recommendations. Known items are excluded by default.
    '''
    def __init__(self, model: Model, buckets, holdouts, ):
        self.model = model
        self.buckets = buckets
        self.holdouts = holdouts
        self.metrics = ["Recall@N"]
        self.model_checkpoints = []
        self.IncrementalTraining_time_record = {}
        self.EvaluateHoldouts_time_record = {}
        self._IncrementalTraining()

    def _IncrementalTraining(self):
        '''
        Incremental training of recommendation model.
        '''
        cold_start_buckets = len( self.buckets ) - len( self.holdouts )
        for b, bucket in enumerate(self.buckets):
            print('bucket', b)
            incrtrain_time = []            
            for i in range(bucket.size):
                uid, iid = bucket.GetTuple(i) # get external IDs
                s = time.time()
                self.model.IncrTrain(uid, iid) # perform incremental training
                f = time.time()
                incrtrain_time.append(f-s)               
            if b >= cold_start_buckets:
                s = time.time()
                self._MakeCheckpoint() # store model
                f = time.time()
                checkpoint_time = f-s
            else:
                checkpoint_time = 0
            self.IncrementalTraining_time_record[f'bucket_{b}'] = {
                'size':bucket.size,
                'train time vector':incrtrain_time,
                'avg train time':np.mean(incrtrain_time),
                'total train time':np.sum(incrtrain_time),
                'checkpoint time':checkpoint_time
            }
    
    def EvaluateHoldouts(self, N_recommendations=20, exclude_known_items:bool=True, default_user:str='none'):
        '''
        exclude_known_items -- boolean, exclude known items from recommendation\n
        default_user -- str. One of: none, random, average, or median.\n\tIf user is not present in model (future user) user factors are generated. If none, then no recommendations are made (user wont count for recall)
        '''
        self.results_matrix = np.zeros( shape=( len( self.holdouts ), len( self.holdouts ) ) )
        metric = self.metrics[0]
        for i, hd in enumerate( self.holdouts ):
            evaluate_time = []
            eh_instance_time = []
            for j, model in enumerate( self.model_checkpoints ):
                eh_instance = EvalHoldout(model=model, holdout=hd, metrics=[metric], N_recommendations=N_recommendations, default_user=default_user)
                s = time.time()
                results = eh_instance.Evaluate(exclude_known_items=exclude_known_items)
                f = time.time()
                evaluate_time.append(f-s)
                result = results[metric]
                del results[metric]
                eh_instance_time.append(results)
                n_not_seen = hd.size - len(result) # if user was not seen, its not added to recall. May be needed to store difference.
                print(f'recommendations not made for users in holdout {i} x bucket {j}: {n_not_seen}')
                result = sum( result ) / len(result)                
                self.results_matrix[i, j] = result
            self.EvaluateHoldouts_time_record[f'holdout_{i}'] = {
                'size': hd.size,
                'train time vector':evaluate_time,
                'avg model eval time':np.mean(evaluate_time),
                'total train time':np.sum(evaluate_time),
                'EvalHoldout time': eh_instance_time
            }
    
    def _MakeCheckpoint(self):
        model_cp = copy.deepcopy(self.model)
        self.model_checkpoints.append(model_cp)
            