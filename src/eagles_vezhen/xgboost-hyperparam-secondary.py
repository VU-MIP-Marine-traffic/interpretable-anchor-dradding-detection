import pickle
import numpy as np
from itertools import product

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_pinball_loss #quantile loss

X_train = np.load('output/data_eagles_vezhen/qrf/X_train.npy')
y_train = np.load('output/data_eagles_vezhen/qrf/y_train.npy')
X_test = np.load('output/data_eagles_vezhen/qrf/X_test.npy')
y_test = np.load('output/data_eagles_vezhen/qrf/y_test.npy')
X_val = np.load('output/data_eagles_vezhen/qrf/X_val.npy')
y_val = np.load('output/data_eagles_vezhen/qrf/y_val.npy')

SLIDINGW_STEPS = 5
N_OUT_FEATS = y_train.shape[-1]
N_INP_FEATS = X_train.shape[-1]

X_train_val = np.concat([X_train, X_val])
y_train_val = np.concat([y_train, y_val])
print(X_train_val.shape)
print(y_train_val.shape)

#merge test and validation to do 5-fold cross-validation
X_train_val = np.concat([X_train, X_val])
y_train_val = np.concat([y_train, y_val])

n_row = X_train_val.shape[0]
n_time = y_train_val.shape[1]
n_inp = X_train_val.shape[2]
n_out = y_train_val.shape[2]

X_train_val = X_train_val.reshape((n_row, n_time*n_inp))
y_train_val = y_train_val.reshape((n_row, n_time*n_out))

param_grid={
    'n_estimators': [128, 64, 32, 16],
    'max_depth': [4, 16],
    'eta': [1e-5, 1e-3, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.25, 0.5],
    'min_child_weight': [1,3,5]
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
        colsample_bytree,
        gamma,
        min_child_weight
    ) = hyperparam_collection

    for ix, (train_index, val_index) in enumerate( KFold(n_splits=5).split(X_train_val) ):

        print(f'Hyperparameter {hyp_ix+1}/{hyperparam_count} | Fold {ix+1}/5')

        X_train_tmp = X_train_val\
                [train_index, :]
        y_train_tmp = y_train_val\
                .reshape((n_row, n_time, n_out))\
                [train_index, 0, 0]
        X_val_tmp = X_train_val\
                [val_index, :]
        y_val_tmp = y_train_val\
                .reshape((n_row, n_time, n_out))\
                [val_index, 0, 0]

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            eta=eta,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            min_child_weight=min_child_weight,
            objective="reg:quantileerror",
            quantile_alpha=0.025,
            device='gpu'
        )
        model.fit(X_train_tmp, y_train_tmp)

        y_pred_low = model.predict(X_val_tmp)
        loss_low = mean_pinball_loss(y_val_tmp, y_pred_low, alpha=0.025)
        if hyperparam_collection not in grid_search_res:
            grid_search_res[hyperparam_collection] = [loss_low]
        else:
            grid_search_res[hyperparam_collection].append( loss_low )

    print(grid_search_res[hyperparam_collection])

with open('output/models/xgboost/hyperparam-first-cog.pkl', 'wb+') as f:
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
        colsample_bytree,
        gamma,
        min_child_weight
    ) = hyperparam_collection

    for ix, (train_index, val_index) in enumerate( KFold(n_splits=5).split(X_train_val) ):

        print(f'Hyperparameter {hyp_ix+1}/{hyperparam_count} | Fold {ix+1}/5')

        X_train_tmp = X_train_val\
                [train_index, :]
        y_train_tmp = y_train_val\
                .reshape((n_row, n_time, n_out))\
                [train_index, 0, 1]
        X_val_tmp = X_train_val\
                [val_index, :]
        y_val_tmp = y_train_val\
                .reshape((n_row, n_time, n_out))\
                [val_index, 0, 1]

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            eta=eta,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            min_child_weight=min_child_weight,
            objective="reg:quantileerror",
            quantile_alpha=0.025,
            device='gpu'
        )
        model.fit(X_train_tmp, y_train_tmp)

        y_pred_low = model.predict(X_val_tmp)
        loss_low = mean_pinball_loss(y_val_tmp, y_pred_low, alpha=0.025)
        if hyperparam_collection not in grid_search_res:
            grid_search_res[hyperparam_collection] = [loss_low]
        else:
            grid_search_res[hyperparam_collection].append( loss_low )

    print(grid_search_res[hyperparam_collection])

with open('output/models/xgboost/hyperparam-first-delta-dif.pkl', 'wb+') as f:
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
        colsample_bytree,
        gamma,
        min_child_weight
    ) = hyperparam_collection

    for ix, (train_index, val_index) in enumerate( KFold(n_splits=5).split(X_train_val) ):

        print(f'Hyperparameter {hyp_ix+1}/{hyperparam_count} | Fold {ix+1}/5')

        X_train_tmp = X_train_val\
                [train_index, :]
        y_train_tmp = y_train_val\
                .reshape((n_row, n_time, n_out))\
                [train_index, 0, 2]
        X_val_tmp = X_train_val\
                [val_index, :]
        y_val_tmp = y_train_val\
                .reshape((n_row, n_time, n_out))\
                [val_index, 0, 2]

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            eta=eta,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            min_child_weight=min_child_weight,
            objective="reg:quantileerror",
            quantile_alpha=0.025,
            device='gpu'
        )
        model.fit(X_train_tmp, y_train_tmp)

        y_pred_low = model.predict(X_val_tmp)
        loss_low = mean_pinball_loss(y_val_tmp, y_pred_low, alpha=0.025)
        if hyperparam_collection not in grid_search_res:
            grid_search_res[hyperparam_collection] = [loss_low]
        else:
            grid_search_res[hyperparam_collection].append( loss_low )

    print(grid_search_res[hyperparam_collection])

with open('output/models/xgboost/hyperparam-first-dif.pkl', 'wb+') as f:
    pickle.dump(grid_search_res, f)




