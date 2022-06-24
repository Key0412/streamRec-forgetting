from eval_implicit import EvalHoldout
from recommenders_implicit import *
import pandas as pd
import numpy as np
import copy

class EvaluateHoldouts():

    def __init__(self, model: Model, buckets, holdouts, N_recommendations=20):
        self.model = model
        self.buckets = buckets
        self.holdouts = holdouts
        self.N_recommendations = N_recommendations
        self.metrics = ["Recall@N"]
        self.model_checkpoints = []
        self._IncrementalTraining()

    def _IncrementalTraining(self):
        '''
        Incremental training of recommendation model.
        '''
        cold_start_buckets = len( self.buckets ) - len( self.holdouts )
        for b, bucket in enumerate(self.buckets):
            print('bucket', b)
            for i in range(bucket.size):
                uid, iid = bucket.GetTuple(i) # get external IDs
                self.model.IncrTrain(uid, iid) # perform incremental training

            if b >= cold_start_buckets:
                self._MakeCheckpoint() # store model
    
    def EvaluateHoldouts(self, exclude_known_items:bool=True, default_user:str='none'):
        '''
        exclude_known_items -- boolean, exclude known items from recommendation
        default_user -- str. One of: random, average, or median. If user is not present in model (new user) user factors are generated.
        '''
        self.results_matrix = np.zeros( shape=( len( self.holdouts ), len( self.holdouts ) ) )
        metric = self.metrics[0]
        for i, hd in enumerate( self.holdouts ):
            for j, model in enumerate( self.model_checkpoints ):
                eh_instance = EvalHoldout(model=model, holdout=hd, metrics=[metric], N_recommendations=self.N_recommendations, default_user=default_user)
                result = sum( eh_instance.Evaluate(exclude_known_items=exclude_known_items)[metric]) / hd.size
                self.results_matrix[i, j] = result
    
    def _MakeCheckpoint(self):
        model_cp = copy.deepcopy(self.model)
        self.model_checkpoints.append(model_cp)
            