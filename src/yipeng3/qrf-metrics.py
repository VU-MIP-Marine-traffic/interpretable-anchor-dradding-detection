import quantile_forest as qrf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_pinball_loss
from itertools import product
import pandas as pd
import numpy as np

def train_and_eval_qrf(X_train, X_test, y_train, y_test, best_hyper):

    qrf_model = qrf.RandomForestQuantileRegressor(**best_hyper, n_jobs=-1)
    qrf_model.fit(X_train, y_train)

    y_pred = qrf_model.predict(X_test, quantiles=[0.025, 0.975])
    y_low = y_pred[:, :, 0]
    y_high = y_pred[:, :, 1]

    y_test =  y_test.reshape((-1,5,3))
    y_low = y_low.reshape((-1,5,3))
    y_high = y_high.reshape((-1,5,3))

    picp_res = np.mean( (y_test < y_high) & (y_test > y_low), axis=0 )
    pinaw_res = np.mean( np.abs(y_high - y_low), axis=0 )

    loss_low = []
    loss_high = []
    for timestep,feature in product(range(5), range(3)):
        loss_low.append(
            mean_pinball_loss(y_test[:, timestep, feature], y_low[:, timestep, feature], alpha=0.025)
        )
        loss_high.append(
            mean_pinball_loss(y_test[:, timestep, feature], y_high[:, timestep, feature], alpha=0.975)
        )
    loss_low = np.array(loss_low).reshape((-1, 3))
    loss_high = np.array(loss_high).reshape((-1, 3))

    df_picp = pd.DataFrame(picp_res, columns=['delta_cog', 'delta_dif', 'dif'])
    df_picp['time_step'] = list(range(5,30,5))
    df_pinaw = pd.DataFrame(pinaw_res, columns=['delta_cog', 'delta_dif', 'dif'])
    df_pinaw['time_step'] = list(range(5,30,5))

    df_loss_low = pd.DataFrame(loss_low, columns=['delta_cog', 'delta_dif', 'dif'])
    df_loss_low['time_step'] = list(range(5,30,5))
    df_loss_high = pd.DataFrame(loss_high, columns=['delta_cog', 'delta_dif', 'dif'])
    df_loss_high['time_step'] = list(range(5,30,5))

    return df_picp, df_pinaw, df_loss_low, df_loss_high

X_train = np.load('output/data_yipeng3/qrf/X_train.npy')
y_train = np.load('output/data_yipeng3/qrf/y_train.npy')
X_test = np.load('output/data_yipeng3/qrf/X_test.npy')
y_test = np.load('output/data_yipeng3/qrf/y_test.npy')
X_val = np.load('output/data_yipeng3/qrf/X_val.npy')
y_val = np.load('output/data_yipeng3/qrf/y_val.npy')

SLIDINGW_STEPS = 5
N_OUT_FEATS = y_train.shape[-1]
N_INP_FEATS = X_train.shape[-1]

y_train = y_train.reshape((-1, SLIDINGW_STEPS*N_OUT_FEATS))
X_train = X_train.reshape((-1, SLIDINGW_STEPS*N_INP_FEATS))
y_val = y_val.reshape((-1, SLIDINGW_STEPS*N_OUT_FEATS))
X_val = X_val.reshape((-1, SLIDINGW_STEPS*N_INP_FEATS))
y_test = y_test.reshape((-1, SLIDINGW_STEPS*N_OUT_FEATS))
X_test = X_test.reshape((-1, SLIDINGW_STEPS*N_INP_FEATS))

best_params = {
    'n_estimators': 100,
    'max_depth': 32,
    'max_samples': 0.75,
    'max_features': 0.75,
    'max_samples_leaf': 5,
}

X = np.concatenate([X_train, X_val, X_test])
y = np.concatenate([y_train, y_val, y_test])
folds = KFold(n_splits=5, random_state=42, shuffle=True)

df_picp = pd.DataFrame()
df_pinaw = pd.DataFrame()
df_loss_low = pd.DataFrame()
df_loss_high = pd.DataFrame()

for ix, (train_idx, test_idx) in enumerate( folds.split(X,y) ):

    X_train = X[train_idx, :]
    y_train = y[train_idx, :]
    X_test = X[test_idx, :]
    y_test = y[test_idx, :]

    print('fold',ix,'shapes:', X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    (
        df_picp_fold,
        df_pinaw_fold,
        df_loss_low_fold,
        df_loss_high_fold,
    ) = train_and_eval_qrf(X_train, X_test, y_train, y_test, best_params)

    df_picp_fold['Fold'] = ix
    df_pinaw_fold['Fold'] = ix
    df_loss_low_fold['Fold'] = ix
    df_loss_high_fold['Fold'] = ix

    df_picp = pd.concat([df_picp, df_picp_fold])
    df_pinaw = pd.concat([df_pinaw, df_pinaw_fold])
    df_loss_low = pd.concat([df_loss_low, df_loss_low_fold])
    df_loss_high = pd.concat([df_loss_high, df_loss_high_fold])

    print(df_loss_low)

df_picp.to_csv('picp-qrf-yp3.csv')
df_pinaw.to_csv('pinaw-qrf-yp3.csv')
df_loss_low.to_csv('loss_low-qrf-yp3.csv')
df_loss_high.to_csv('loss_high-qrf-yp3.csv')
