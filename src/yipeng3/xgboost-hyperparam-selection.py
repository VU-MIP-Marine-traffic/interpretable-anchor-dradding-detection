"""
Requires:
- output/data_yipeng3/qrf/X_train.npy
- output/data_yipeng3/qrf/y_train.npy
- output/data_yipeng3/qrf/X_val.npy
- output/data_yipeng3/qrf/y_val.npy
"""

import os
import pickle
import numpy as np
from itertools import product

import xgboost as xgb
from sklearn.metrics import mean_pinball_loss #quantile loss

X_train = np.load('output/data_yipeng3/qrf/X_train.npy')
y_train = np.load('output/data_yipeng3/qrf/y_train.npy')
X_val = np.load('output/data_yipeng3/qrf/X_val.npy')
y_val = np.load('output/data_yipeng3/qrf/y_val.npy')

SLIDINGW_STEPS = 5
N_OUT_FEATS = y_train.shape[-1]
N_INP_FEATS = X_train.shape[-1]

n_time = y_train.shape[1]
n_inp = X_train.shape[2]
n_out = y_train.shape[2]

X_train = X_train.reshape((-1, n_time*n_inp))
y_train = y_train.reshape((-1, n_time*n_out))

X_val = X_val.reshape((-1, n_time*n_inp))
y_val = y_val.reshape((-1, n_time*n_out))

param_grid={
    'n_estimators': [32, 64, 128],
    'max_depth': [4, 16],
    'eta': [0.05, 0.01, 0.005, 0.001],
    'subsample': [0.6, 0.8, 1.0],
}

hyperparam_count = len( list( product(*param_grid.values()) ) )
grid_search_res: dict = {}

for hyp_ix, hyperparam_collection in enumerate( list(product( *param_grid.values() )) ):

    print(hyperparam_collection)
    (
        n_estimators,
        max_depth,
        eta,
        subsample,
    ) = hyperparam_collection

    print(f'Hyperparameter {hyp_ix+1}/{hyperparam_count}')

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        eta=eta,
        subsample=subsample,
        objective="reg:quantileerror",
        quantile_alpha=0.025,
        device='gpu'
    )
    model.fit(X_train, y_train[:, 0])

    y_pred_low = model.predict(X_val)
    loss_low = mean_pinball_loss(y_val[:, 0], y_pred_low, alpha=0.025)
    if hyperparam_collection not in grid_search_res:
        grid_search_res[hyperparam_collection] = [loss_low]
    else:
        grid_search_res[hyperparam_collection].append( loss_low )

    print(grid_search_res[hyperparam_collection])

os.makedirs('output/models/xgboost/', exists_ok=True)
with open('output/models/xgboost/hyperparam-first-cog-yipeng3.pkl', 'wb+') as f:
    pickle.dump(grid_search_res, f)


hyperparam_count = len( list( product(*param_grid.values()) ) )
grid_search_res: dict = {}


for hyp_ix, hyperparam_collection in enumerate( list(product( *param_grid.values() )) ):

    print(hyperparam_collection)
    (
        n_estimators,
        max_depth,
        eta,
        subsample,
    ) = hyperparam_collection

    print(f'Hyperparameter {hyp_ix+1}/{hyperparam_count}')

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        eta=eta,
        subsample=subsample,
        objective="reg:quantileerror",
        quantile_alpha=0.025,
        device='gpu'
    )
    model.fit(X_train, y_train[:, 1])

    y_pred_low = model.predict(X_val)
    loss_low = mean_pinball_loss(y_val[:, 1], y_pred_low, alpha=0.025)
    if hyperparam_collection not in grid_search_res:
        grid_search_res[hyperparam_collection] = [loss_low]
    else:
        grid_search_res[hyperparam_collection].append( loss_low )

    print(grid_search_res[hyperparam_collection])

with open('output/models/xgboost/hyperparam-first-delta-dif-yipeng3.pkl', 'wb+') as f:
    pickle.dump(grid_search_res, f)


hyperparam_count = len( list( product(*param_grid.values()) ) )
grid_search_res: dict = {}


for hyp_ix, hyperparam_collection in enumerate( list(product( *param_grid.values() )) ):

    print(hyperparam_collection)
    (
        n_estimators,
        max_depth,
        eta,
        subsample,
    ) = hyperparam_collection

    print(f'Hyperparameter {hyp_ix+1}/{hyperparam_count}')

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        eta=eta,
        subsample=subsample,
        objective="reg:quantileerror",
        quantile_alpha=0.025,
        device='gpu'
    )
    model.fit(X_train, y_train[:, 2])

    y_pred_low = model.predict(X_val)
    loss_low = mean_pinball_loss(y_val[:, 2], y_pred_low, alpha=0.025)
    if hyperparam_collection not in grid_search_res:
        grid_search_res[hyperparam_collection] = [loss_low]
    else:
        grid_search_res[hyperparam_collection].append( loss_low )

    print(grid_search_res[hyperparam_collection])

with open('output/models/xgboost/hyperparam-first-dif-yipeng3.pkl', 'wb+') as f:
    pickle.dump(grid_search_res, f)
