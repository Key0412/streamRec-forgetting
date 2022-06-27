from .implicit_data import ImplicitData

import pandas as pd

def getBucketsHoldouts(data:pd.DataFrame, user_col:str, item_col:str, frequent_users:list, interval_type:str=None, intervals:list=None, cold_start_buckets:int=1):
    '''
    Creates lists with buckets and holdouts based on passed intervals.
    
    data - interactions, must contain 'date' column\n
    user_col - name of column with user IDs\n
    item_col - name of column with item IDs\n
    frequent_users - list of frequent users\n
    interval_type - M for month, QS for quarter or semester, F representing fixed bucket size\n
    intervals - list containing tuple intervals. pos0-interval start, pos1-interval end. for QS these are dates, for F these are indexes. not necessary for Month interval type.\n
    cold_start_buckets - number of buckets to be used for training only\n
    '''
    buckets = []
    assert interval_type in ['M', 'QS', 'F'], "interval must be one of M, QS, or F"
    if interval_type == 'M':
        # create buckets based on months
        months = data['date'].unique()
        for interval in months:
            idx = (data['date'] == interval)
            buckets.append( data[idx] )
    elif interval_type == 'QS':
        # create buckets based on quarters or semesters
        for s, e in intervals:
            idx = (data['date'] >= s) & (data['date'] <= e)
            buckets.append( data[idx] )
        else:
            idx = (data['date'] > e)
            buckets.append( data[idx] )
    else:
        # create buckets based on fixed number of examples
        for i, j in intervals:
            buckets.append( data.iloc[i:j] )

    # create holdouts with last user interaction
    holdouts = []
    frequent_users_seen = [] # frequent users must have been seen at least once before being sent to holdouts. 
    # Imagine if the first frequent user interaction is the single interaction by this user in an interval, then this single interaction cant be sent to the holdout.
    for i, b in enumerate( buckets ):
        if i >= cold_start_buckets:
            last_interaction_idx = []
            for u in frequent_users:
                idx = b[user_col] == u
                if (idx.sum() == 1) and (u not in frequent_users_seen): # first condition to see if user appears once, second to see if user was not seen before - then it wont go to holdout, and it will be marked as seen
                    frequent_users_seen.append(u)
                    continue
                elif idx.sum() > 0: # else, if user appears at least once, append index to holdout
                    last_interaction_idx.append( b[ idx ].index[-1] )
                    if (u not in frequent_users_seen): # and if user hasnt been seen, mark as seen (he must appear at least twice then)
                        frequent_users_seen.append(u)
            holdout = b.loc[ last_interaction_idx ] # get last interactions as holdout
            holdout.reset_index(drop=True, inplace=True) # reset index required - implicitdata indexes user by their previous index
            holdout = ImplicitData(user_list=holdout[user_col], item_list=holdout[item_col]) # convert holdout to ImplicitData
            holdouts.append(holdout) # append to holdouts
            bucket = b.drop( index = last_interaction_idx) # remove last interactions from bucket
        else: # if bucket belongs to 'cold_start_buckets'
            bucket = b
            for u in frequent_users: # as before, we mark frequent users in the cold start bucket as seen
                idx = b[user_col] == u
                if (idx.sum() > 0):
                    frequent_users_seen.append(u)
        bucket.reset_index(drop=True, inplace=True) # reset index required - implicitdata indexes user by their previous index
        buckets[i] = ImplicitData(user_list=bucket[user_col], item_list=bucket[item_col]) # convert bucket to ImplicitData and store

    return buckets, holdouts