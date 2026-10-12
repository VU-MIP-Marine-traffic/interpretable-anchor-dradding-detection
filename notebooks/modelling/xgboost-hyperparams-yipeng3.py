# Check which hyperparameters were best after running XGBoost hyperparameter optimisation for Yi Peng 3 data

# Refactored with assistance from OpenAI ChatGPT (GPT-6).
# Changes made: Removed Jupyter/nbconvert cell markers and empty cells,
# removed unused matplotlib and numpy imports, and converted notebook
# display expressions into explicit print() calls. Preserved the original
# data processing, calculations, and input paths.

import pandas as pd
import pickle


with open('../../output/models/xgboost/hyperparam-first-cog-yipeng3.pkl', 'rb+') as f:
    hyperparam_cog = pickle.load(f)

with open('../../output/models/xgboost/hyperparam-first-delta-dif-yipeng3.pkl', 'rb+') as f:
    hyperparam_delta_dif = pickle.load(f)

with open('../../output/models/xgboost/hyperparam-first-dif-yipeng3.pkl', 'rb+') as f:
    hyperparam_dif = pickle.load(f)


HYPERPARAMS = [
    'n_estimators',
    'max_depth',
    'eta',
    'subsample'
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


df = pd.concat([df_cog, df_delta_dif, df_dif])

df_hyper = df.groupby(by=HYPERPARAMS + ['var'])\
    .agg(['mean', 'std'])\
    ['loss']\
    .reset_index()


min_mean_var = df_hyper.groupby('var')['mean'].idxmin().tolist()

print('\nBest hyperparams for delta COG\n', df_hyper.iloc[min_mean_var[0], :].to_dict())

print('\nBest hyperparams for delta DIF\n', df_hyper.iloc[min_mean_var[1], :].to_dict())

print('\nBest hyperparams for DIF\n', df_hyper.iloc[min_mean_var[2], :].to_dict())
