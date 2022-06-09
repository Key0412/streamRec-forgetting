from eval_implicit import EvaluateAndStore
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')


def lineplot_recallxholdout(df,
    title='Recall@20 for checkpoint models across Holdouts - model - data',
    filepath='images/lineplots/..'):
    
    plt.figure(figsize=(25,10))
    sns.lineplot(data=df.T, palette='tab20')
    x_t = np.arange(0,20)
    plt.xticks(x_t, labels=[str(i+1) for i in x_t])
    plt.xlim(0, 19)
    plt.xlabel('Holdout')
    plt.ylabel('Recall@20')
    plt.legend(bbox_to_anchor=(1.0,1), loc="upper left", labels=[f'model: T{model+1}' for model in range( df.shape[0] )])
    plt.title(title)
    if filepath:
        plt.savefig(filepath);

def recall_heatmap(df,
    title='Recall@20 for checkpoint models across Holdouts - model - data',
    filepath='images/heatmaps/..'):
    plt.figure(figsize=(15, 10))
    x_t = np.arange(0,20)
    labels=[str(i+1) for i in x_t]
    sns.heatmap(df, vmin=0, vmax=df.max().max(), annot=True, fmt='0.2f', linewidths=.1, cmap='Spectral_r', xticklabels=labels, yticklabels=labels)
    plt.ylabel('model')
    plt.xlabel('holdout')
    plt.title(title)
    if filepath:
        plt.savefig(filepath);

def plot_n_users_per_bucket(eval_object:EvaluateAndStore, dataset_name:str, filename:str=None):
    n_users = len( eval_object.data.userset )
    n_users_bucket = pd.Series( [len( bucket.userset ) for bucket in eval_object.holdouts] )
    n_users_bucket = n_users_bucket.reset_index()
    n_users_bucket.columns = ['Bucket', 'N_users']
    n_users_bucket['Bucket'] = n_users_bucket['Bucket']+1
    plt.figure(figsize=(10,5))
    sns.barplot(x='Bucket', y='N_users', data=n_users_bucket, color='b', label='users per bucket')
    sns.lineplot(data=np.repeat(n_users, n_users_bucket.shape[0]), label='total users', color='orange')
    plt.title(f'Users per bucket - {dataset_name}');
    if filename:
        plt.savefig(f'images/user_bucket_analysis/{filename}')