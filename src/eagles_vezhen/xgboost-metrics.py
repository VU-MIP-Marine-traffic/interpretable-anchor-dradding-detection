import pandas as pd
import numpy as np
import quantile_forest as qrf
from sklearn.model_selection import KFold

X_train = np.load('X_train_final.npy')
y_train = np.load('y_train_final.npy')

X_val = np.load('X_val_final.npy')
y_val = np.load('y_val_final.npy')

SLIDINGW_STEPS = 5
N_OUT_FEATS = y_train.shape[-1]
N_INP_FEATS = X_train.shape[-1]

X_train_val = np.concat([X_train, X_val])
y_train_val = np.concat([y_train, y_val])

params = {
    'max_depth': 16,
    'max_samples': 0.75,
    'max_samples_leaf': 1,
    'max_features': 0.75
}

#merge test and validation to do 5-fold cross-validation
X_train_val = np.concat([X_train, X_val])
y_train_val = np.concat([y_train, y_val])

n_row = X_train_val.shape[0]
n_time = y_train_val.shape[1]
n_inp = X_train_val.shape[2]
n_out = y_train_val.shape[2]

X_train_val = X_train_val.reshape((n_row, n_time*n_inp))
y_train_val = y_train_val.reshape((n_row, n_time*n_out))

results = {
    'picp_delta_cog': [],
    'pinaw_delta_cog': [],
    'picp_delta_dif': [],
    'pinaw_delta_dif': []
}

fold_obj = KFold(n_splits=5, random_state=0, shuffle=True).split(X_train_val)
for ix, (train_index, val_index) in enumerate(fold_obj):

    print('fold', ix+1, 'of 5')

    X_train_tmp = X_train_val\
            [train_index, :]
    y_train_tmp = y_train_val[train_index, :]

    X_val_tmp = X_train_val\
            [val_index, :]
    y_val_tmp = y_train_val[val_index, :]

    qrf_model = qrf.RandomForestQuantileRegressor(
        **params,
        n_jobs=-1,
        verbose=True,
        random_state=0
    )
    qrf_model.fit(X_train_tmp, y_train_tmp)

    y_pred_low = qrf_model.predict(X_val_tmp, quantiles=[0.025])
    y_pred_high = qrf_model.predict(X_val_tmp, quantiles=[0.975])

    in_pi = (y_pred_low < y_val_tmp) & (y_pred_high > y_val_tmp)
    in_pi = in_pi.reshape((-1, 25, 2))
    in_pi_delta_cog = in_pi[:, :, 0]
    in_pi_delta_dif = in_pi[:, :, 1]

    width_pi =  np.abs(y_pred_high - y_pred_low)
    width_pi = width_pi.reshape((-1, 25, 2))
    width_pi_delta_cog = width_pi[:, :, 0]
    width_pi_delta_dif = width_pi[:, :, 1]

    results['picp_delta_cog'].append(np.mean(in_pi_delta_cog))
    results['picp_delta_dif'].append(np.mean(in_pi_delta_dif))
    results['pinaw_delta_cog'].append(np.mean(width_pi_delta_cog))
    results['pinaw_delta_dif'].append(np.mean(width_pi_delta_dif))

    print( np.mean(in_pi_delta_cog) )
    print( np.mean(in_pi_delta_dif) )
    print( np.mean(width_pi_delta_cog) )
    print( np.mean(width_pi_delta_dif) )

print(results)
pd.DataFrame(results).to_csv('-xgb-eagles.csv')
