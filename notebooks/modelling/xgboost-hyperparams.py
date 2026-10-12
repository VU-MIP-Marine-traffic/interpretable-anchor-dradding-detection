# Check which hyperparameters were best using after running xgboost hyperparameter optimisation

# Refactored with assistance from OpenAI ChatGPT (GPT-6).
# Changes made: Removed Jupyter/nbconvert cell markers and empty cells,
# removed the unused matplotlib import, and converted notebook display
# expressions into explicit print() calls. Preserved the original data
# processing, calculations, input paths, and source attributions.

import pandas as pd
import pickle


with open('../../output/models/xgboost/hyperparam-first-cog.pkl', 'rb+') as f:
    hyperparam_cog = pickle.load(f)

with open('../../output/models/xgboost/hyperparam-first-delta-dif.pkl', 'rb+') as f:
    hyperparam_delta_dif = pickle.load(f)

with open('../../output/models/xgboost/hyperparam-first-dif.pkl', 'rb+') as f:
    hyperparam_dif = pickle.load(f)


HYPERPARAMS = [
    'n_estimators',
    'max_depth',
    'eta',
    'subsample',
]

df_cog = pd.DataFrame(hyperparam_cog).melt()
df_cog.columns = HYPERPARAMS + ['loss']
df_cog['var'] = 'delta_COG'

df_delta_dif = pd.DataFrame(hyperparam_delta_dif).melt()
df_delta_dif.columns = HYPERPARAMS + ['loss']
df_delta_dif['var'] = 'delta_dif'

df_dif = pd.DataFrame(hyperparam_dif).melt()
df_dif.columns = HYPERPARAMS + ['loss']
df_dif['var'] = 'dif'


print('Best hyperparams for delta DIF\n', df_delta_dif.iloc[df_delta_dif['loss'].argmin(), :])

print('\nBest hyperparams for DIF\n', df_dif.iloc[df_dif['loss'].argmin(), :])

print('\nBest hyperparams for delta COG\n', df_cog.iloc[df_cog['loss'].argmin(), :])
