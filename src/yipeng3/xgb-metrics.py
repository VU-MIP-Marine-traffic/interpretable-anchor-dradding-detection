import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_pinball_loss
import pandas as pd
import numpy as np

def train_and_eval_qrf(X_train, X_test, y_train, y_test, best_hyper):


    picp_res = []
    pinaw_res = []
    loss_low = []
    loss_high = []

    for feat in range(3):
        for t in range(5):

            y_train_tmp = y_train\
                .reshape((-1, 5, 3))\
                [:, t, feat]
            y_test_tmp = y_test\
                .reshape((-1, 5, 3))\
                [:, t, feat]

            model_low = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=0.025,
                device='gpu',
                random_state=0,
                **best_hyper[feat]
            )

            model_high = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=0.975,
                device='gpu',
                random_state=0,
                **best_hyper[feat]
            )

            model_low.fit(X_train, y_train_tmp)
            model_high.fit(X_train, y_train_tmp)

            y_low = model_low.predict(X_test)
            y_high = model_high.predict(X_test)

            picp_res.append(np.mean( (y_test_tmp < y_high) & (y_test_tmp > y_low) ))
            pinaw_res.append(np.mean( np.abs(y_high - y_low) ))

            loss_low.append(mean_pinball_loss(y_test_tmp, y_low, alpha=0.025))
            loss_high.append(mean_pinball_loss(y_test_tmp, y_high, alpha=0.975))

    print( picp_res )
    df_picp = pd.DataFrame(picp_res, columns=['picp'])
    df_picp['time_step'] = list(range(5,30,5))*3
    df_picp['feature'] = ['delta_cog']*5 + ['delta_dif']*5 + ['dif']*5

    df_pinaw = pd.DataFrame(pinaw_res, columns=['pinaw'])
    df_pinaw['time_step'] = list(range(5,30,5))*3
    df_pinaw['feature'] = ['delta_cog']*5 + ['delta_dif']*5 + ['dif']*5

    df_loss_low = pd.DataFrame(loss_low, columns=['loss_low'])
    df_loss_low['time_step'] = list(range(5,30,5))*3
    df_loss_low['feature'] = ['delta_cog']*5 + ['delta_dif']*5 + ['dif']*5

    df_loss_high = pd.DataFrame(loss_low, columns=['loss_low'])
    df_loss_high['time_step'] = list(range(5,30,5))*3
    df_loss_high['feature'] = ['delta_cog']*5 + ['delta_dif']*5 + ['dif']*5

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

# different params for each feature
best_params = [
    # delta cog
    {
        'n_estimators': 128,
        'max_depth': 4,
        'eta': 0.05,
        'subsample': 0.6,
    },
    # delta dif
    {
        'n_estimators': 128,
        'max_depth': 4,
        'eta': 0.05,
        'subsample': 1.0,
    },
    # dif
    {
        'n_estimators': 128,
        'max_depth': 4,
        'eta': 0.05,
        'subsample': 1.0,
    },
]

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

df_picp.to_csv('picp-xgb-yp3.csv')
df_pinaw.to_csv('pinaw-xgb-yp3.csv')
df_loss_low.to_csv('loss_low-xgb-yp3.csv')
df_loss_high.to_csv('loss_high-xgb-yp3.csv')
