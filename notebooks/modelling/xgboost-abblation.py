# Run from notebooks/modelling/

import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from pathlib import Path
from datetime import datetime, timedelta


IMAGE_DIR = Path("../images")
OUTPUT_DIR = Path("../../output")

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


X_train = np.load('../../output/data_eagles_vezhen/qrf/X_train2.npy')
y_train = np.load('../../output/data_eagles_vezhen/qrf/y_train2.npy')
X_test = np.load('../../output/data_eagles_vezhen/qrf/X_test2.npy')
y_test = np.load('../../output/data_eagles_vezhen/qrf/y_test2.npy')
X_val = np.load('../../output/data_eagles_vezhen/qrf/X_val2.npy')
y_val = np.load('../../output/data_eagles_vezhen/qrf/y_val2.npy')

SLIDINGW_STEPS = 5
N_OUT_FEATS = y_train.shape[-1]
N_INP_FEATS = X_train.shape[-1]

y_train = y_train.reshape((-1, SLIDINGW_STEPS * N_OUT_FEATS))
X_train = X_train.reshape((-1, SLIDINGW_STEPS * N_INP_FEATS))
y_val = y_val.reshape((-1, SLIDINGW_STEPS * N_OUT_FEATS))
X_val = X_val.reshape((-1, SLIDINGW_STEPS * N_INP_FEATS))
y_test = y_test.reshape((-1, SLIDINGW_STEPS * N_OUT_FEATS))
X_test = X_test.reshape((-1, SLIDINGW_STEPS * N_INP_FEATS))


vezhen_y = np.load(
    '../../output/data_eagles_vezhen/qrf/vezhen_y3.npy',
    allow_pickle=True
).reshape((-1, 5 * 3))

vezhen_x = np.load(
    '../../output/data_eagles_vezhen/qrf/vezhen_x3.npy',
    allow_pickle=True
).reshape((-1, 5 * 20))

eagles_y = np.load(
    '../../output/data_eagles_vezhen/qrf/eagles_y3.npy',
    allow_pickle=True
).reshape((-1, 5 * 3))

eagles_x = np.load(
    '../../output/data_eagles_vezhen/qrf/eagles_x3.npy',
    allow_pickle=True
).reshape((-1, 5 * 20))



def test_xgboost_feature(
    feat: int, feat_name: str, hyper: dict,
    X_train, y_train, X_val, y_val,
    eagles_x, eagles_y, vezhen_x, vezhen_y
):

    in_pi_mask = None
    eagles_low = None
    eagles_high = None
    vezhen_low = None
    vezhen_high = None

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

        eagles_low_slice = model_low.predict(eagles_x)
        eagles_high_slice = model_high.predict(eagles_x)
        vezhen_low_slice = model_low.predict(vezhen_x)
        vezhen_high_slice = model_high.predict(vezhen_x)

        y_low = model_low.predict(X_val)
        y_high = model_high.predict(X_val)

        val_mmsis = np.load(
            '../../output/data_eagles_vezhen/qrf/val-mmsis2.npy',
            allow_pickle=True
        )
        val_days = np.load(
            '../../output/data_eagles_vezhen/qrf/val-days2.npy',
            allow_pickle=True
        )

        in_pi_mask_slice = (y_val_tmp > y_low) & (y_val_tmp < y_high)

        if t == 0:
            in_pi_mask = in_pi_mask_slice.reshape((-1, 1))
            eagles_low = eagles_low_slice.reshape((-1, 1))
            eagles_high = eagles_high_slice.reshape((-1, 1))
            vezhen_low = vezhen_low_slice.reshape((-1, 1))
            vezhen_high = vezhen_high_slice.reshape((-1, 1))
        else:
            in_pi_mask = np.concatenate(
                [in_pi_mask, in_pi_mask_slice.reshape((-1, 1))],
                axis=1
            )
            eagles_low = np.concatenate(
                [eagles_low, eagles_low_slice.reshape((-1, 1))],
                axis=1
            )
            eagles_high = np.concatenate(
                [eagles_high, eagles_high_slice.reshape((-1, 1))],
                axis=1
            )
            vezhen_low = np.concatenate(
                [vezhen_low, vezhen_low_slice.reshape((-1, 1))],
                axis=1
            )
            vezhen_high = np.concatenate(
                [vezhen_high, vezhen_high_slice.reshape((-1, 1))],
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

    eagles_y_tmp = eagles_y.reshape((-1, 5, 3))[:, :, feat]
    print(
        'EAGLE S',
        feat_name,
        np.mean(
            (eagles_y_tmp < eagles_high) &
            (eagles_y_tmp > eagles_low)
        )
    )

    vezhen_y_tmp = vezhen_y.reshape((-1, 5, 3))[:, :, feat]
    print(
        'VEZHEN',
        feat_name,
        np.mean(
            (vezhen_y_tmp < vezhen_high) &
            (vezhen_y_tmp > vezhen_low)
        )
    )


best_hyper = {
    'Delta COG': {
        'n_estimators': 128,
        'max_depth': 4,
        'eta': 0.05,
        'subsample': 0.8,
    },

    # This is NOT a typo - the parameters sets were the same for both variables

    'Delta DIF': {
        'n_estimators': 128,
        'max_depth': 4,
        'eta': 0.05,
        'subsample': 0.8,
    },
    'DIF': {
        'n_estimators': 128,
        'max_depth': 4,
        'eta': 0.05,
        'subsample': 1.0,
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
        eagles_x=eagles_x,
        eagles_y=eagles_y,
        vezhen_x=vezhen_x,
        vezhen_y=vezhen_y,
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
    print( eagles_x.shape )

    X_train_subset = X_train[:, keep_feature_mask]
    X_val_subset = X_val[:, keep_feature_mask]
    eagles_x_subset = eagles_x[:, keep_feature_mask]
    vezhen_x_subset = vezhen_x[:, keep_feature_mask]

    test_xgboost_feature(
        ix,
        feat_name=var_name,
        hyper=hyperparams,
        X_train=X_train_subset,
        X_val=X_val_subset,
        y_train=y_train,
        y_val=y_val,
        eagles_x=eagles_x_subset,
        eagles_y=eagles_y,
        vezhen_x=vezhen_x_subset,
        vezhen_y=vezhen_y,
    )
