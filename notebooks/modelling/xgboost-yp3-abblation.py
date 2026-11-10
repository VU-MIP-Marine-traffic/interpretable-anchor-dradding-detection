
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from pathlib import Path
from datetime import datetime, timedelta

plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'

IMAGE_DIR = Path("../images")
DATA_DIR = Path("../../output/data_yipeng3/qrf")

IMAGE_DIR.mkdir(parents=True, exist_ok=True)


X_train = np.load(DATA_DIR / 'X_train.npy')
y_train = np.load(DATA_DIR / 'y_train.npy')
X_test = np.load(DATA_DIR / 'X_test.npy')
y_test = np.load(DATA_DIR / 'y_test.npy')
X_val = np.load(DATA_DIR / 'X_val.npy')
y_val = np.load(DATA_DIR / 'y_val.npy')

SLIDINGW_STEPS = 5
N_OUT_FEATS = y_train.shape[-1]
N_INP_FEATS = X_train.shape[-1]

y_train = y_train.reshape((-1, SLIDINGW_STEPS * N_OUT_FEATS))
X_train = X_train.reshape((-1, SLIDINGW_STEPS * N_INP_FEATS))
y_val = y_val.reshape((-1, SLIDINGW_STEPS * N_OUT_FEATS))
X_val = X_val.reshape((-1, SLIDINGW_STEPS * N_INP_FEATS))
y_test = y_test.reshape((-1, SLIDINGW_STEPS * N_OUT_FEATS))
X_test = X_test.reshape((-1, SLIDINGW_STEPS * N_INP_FEATS))

yipeng3_y = np.load(
    DATA_DIR / 'yipeng3_y.npy',
    allow_pickle=True
).reshape((-1, 5 * 3))
yipeng3_x = np.load(
    DATA_DIR / 'yipeng3_x.npy',
    allow_pickle=True
).reshape((-1, 5 * 20))

val_mmsis = np.load(DATA_DIR / 'val-mmsis.npy', allow_pickle=True)
val_days = np.load(DATA_DIR / 'val-days.npy', allow_pickle=True)
test_mmsis = np.load(DATA_DIR / 'test-mmsis.npy', allow_pickle=True)
test_days = np.load(DATA_DIR / 'test-days.npy', allow_pickle=True)


def test_xgboost_feature(
    feat: int, feat_name: str, hyper: dict,
    X_train, y_train, X_val, y_val,
    yipeng3_x, yipeng3_y
):

    in_pi_mask = None
    yipeng3_low = None
    yipeng3_high = None

    for t in range(5):
        y_train_tmp = y_train.reshape((-1, 5, 3))[:, t, feat]
        y_val_tmp = y_val.reshape((-1, 5, 3))[:, t, feat]

        model_low = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.025,
            device='gpu',
            **hyper
        )

        model_high = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.975,
            device='gpu',
            **hyper
        )

        feat = 0

        model_low.fit(X_train, y_train_tmp)
        model_high.fit(X_train, y_train_tmp)

        yipeng3_low_slice = model_low.predict(yipeng3_x)
        yipeng3_high_slice = model_high.predict(yipeng3_x)

        y_low = model_low.predict(X_val)
        y_high = model_high.predict(X_val)

        in_pi_mask_slice = (y_val_tmp > y_low) & (y_val_tmp < y_high)

        if t == 0:
            in_pi_mask = in_pi_mask_slice.reshape((-1, 1))
            yipeng3_low = yipeng3_low_slice.reshape((-1, 1))
            yipeng3_high = yipeng3_high_slice.reshape((-1, 1))
        else:
            in_pi_mask = np.concatenate(
                [in_pi_mask, in_pi_mask_slice.reshape((-1, 1))],
                axis=1
            )
            yipeng3_low = np.concatenate(
                [yipeng3_low, yipeng3_low_slice.reshape((-1, 1))],
                axis=1
            )
            yipeng3_high = np.concatenate(
                [yipeng3_high, yipeng3_high_slice.reshape((-1, 1))],
                axis=1
            )

    group_results = {}

    for ix, (mmsi, day) in enumerate(zip(val_mmsis, val_days)):
        if (mmsi, day) not in group_results:
            group_results[(mmsi, day)] = [in_pi_mask[ix, :]]
        else:
            group_results[(mmsi, day)].append(in_pi_mask[ix, :])

    df = []

    for (mmsi, day), pi_masks in group_results.items():
        means = np.mean(pi_masks)
        df.append({
            'mmsi': mmsi,
            'day': day,
            'delta_cog': means
        })

    df = pd.DataFrame(df)

    print(feat_name, ':', np.quantile(df['delta_cog'], 0.05))

    yipeng3_y_tmp = yipeng3_y.reshape((-1, 5, 3))[:, :, feat]
    print(
        'Yipeng-3',
        feat_name,
        np.mean(
            (yipeng3_y_tmp < yipeng3_high) &
            (yipeng3_y_tmp > yipeng3_low)
        )
    )

best_hyper = {
    'Delta COG': {
        'n_estimators': 128,
        'max_depth': 4,
        'eta': 0.05,
        'subsample': 0.75
    },

    # This is NOT a typo - the parameters sets were the same for both variables

    'Delta DIF': {
        'n_estimators': 128,
        'max_depth': 4,
        'eta': 0.05,
        'subsample': 0.75,
    },
    'DIF': {
        'n_estimators': 128,
        'max_depth': 4,
        'eta': 0.05,
        'subsample': 1.0
    }
}

print('With weather variables:')

for ix, (var_name, hyperparams) in enumerate( best_hyper.items() ):

    test_xgboost_feature(
        ix,
        feat_name=var_name,
        hyper=hyperparams,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        yipeng3_x=yipeng3_x,
        yipeng3_y=yipeng3_y,
    )


features = [
    'SPEED', 'COURSE', 'currentSpeed', 'gust', 'swellHeight',
    'waveHeight', 'windSpeed', 'dif', 'COURSE-dif', 'delta_dif',
    'COURSE_vec_x', 'COURSE_vec_y', 'currentDirection_vec_x',
    'currentDirection_vec_y', 'swellDirection_vec_x',
    'swellDirection_vec_y', 'waveDirection_vec_x', 'waveDirection_vec_y',
    'windDirection_vec_x', 'windDirection_vec_y'
]
meteo_features = [
    'currentDirection_vec_x', 'currentDirection_vec_y',
    'swellDirection_vec_x',   'swellDirection_vec_y',
    'waveDirection_vec_x',    'waveDirection_vec_y',
    'windDirection_vec_x',    'windDirection_vec_y'
]

feature_names = [
    rf'${col}(t-{t})$' if t != 0 else f'${col}(t)$'
    for t in range(0, 25, 5)
    for col in features
]
keep_feature_mask = [
    any([meteo_feat in feature for meteo_feat in meteo_features])
    for feature in feature_names
]

print('Without weather variables:')

for ix, (var_name, hyperparams) in enumerate( best_hyper.items() ):

    print( X_train.shape )
    print( X_val.shape )

    X_train_subset = X_train[:, keep_feature_mask]
    X_val_subset = X_val[:, keep_feature_mask]
    yipeng3_x_subset = yipeng3_x[:, keep_feature_mask]

    test_xgboost_feature(
        ix,
        feat_name=var_name,
        hyper=hyperparams,
        X_train=X_train_subset,
        X_val=X_val_subset,
        y_train=y_train,
        y_val=y_val,
        yipeng3_x=yipeng3_x_subset,
        yipeng3_y=yipeng3_y,
    )
