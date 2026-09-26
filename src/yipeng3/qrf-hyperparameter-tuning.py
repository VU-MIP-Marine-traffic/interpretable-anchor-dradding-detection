import time
import pickle
import numpy as np
from itertools import product

import quantile_forest as qrf
from sklearn.model_selection import KFold
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
    'max_samples_leaf': [3, 5],
    'max_features': [0.75, 'sqrt'],
    'max_samples': [0.5, 0.75, 1.0],
    'max_depth': [4, 32, 64]
}

hyperparam_count = len( list( product(*param_grid.values()) ) )
grid_search_res: dict = {}

for hyp_ix, hyperparam_collection in enumerate( list(product( *param_grid.values() )) ):

    print(hyperparam_collection)
    (
        max_samples_leaf,
        max_features,
        max_samples,
        max_depth
    ) = hyperparam_collection

    print(f'Hyperparameter {hyp_ix+1}/{hyperparam_count}')

    model = qrf.RandomForestQuantileRegressor(
        max_samples_leaf=max_samples_leaf,
        max_features=max_features,
        max_depth=max_depth,
        max_samples=max_samples,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred_low = model.predict(X_val, quantiles=[0.025])
    y_pred_high = model.predict(X_val, quantiles=[0.975])

    loss_low = mean_pinball_loss(y_val, y_pred_low, alpha=0.025)
    loss_high = mean_pinball_loss(y_val, y_pred_high, alpha=0.975)

    print(loss_low)
    print(loss_high)

    if hyperparam_collection not in grid_search_res:
        grid_search_res[hyperparam_collection] = [(loss_low, loss_high)]
    else:
        grid_search_res[hyperparam_collection].append( (loss_low, loss_high) )

with open('qrf-hyperparam.pkl', 'wb+') as f:
    pickle.dump(grid_search_res, f)
