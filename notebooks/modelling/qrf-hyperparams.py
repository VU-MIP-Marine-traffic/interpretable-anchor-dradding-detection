import pandas as pd
import pickle


with open('../../output/models/qrf/hyperparam-fifth.pkl', 'rb+') as f:
    hyperparam_qrf = pickle.load(f)

HYPERPARAMS = [
    'max_samples_leaf',
    'max_features',
    'max_samples',
    'max_depth'
]

df_qrf = pd.DataFrame(hyperparam_qrf).melt()
df_qrf.columns = HYPERPARAMS + ['loss']

print('\nBest hyperparams for qrf\n', df_qrf.iloc[df_qrf['loss'].argmin(), :])
