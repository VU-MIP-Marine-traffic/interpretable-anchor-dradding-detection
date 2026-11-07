import numpy as np
import pickle
import pandas as pd
import quantile_forest as qrf


X_train = np.load('output/data_yipeng3/qrf/X_train.npy')
y_train = np.load('output/data_yipeng3/qrf/y_train.npy')

SLIDINGW_STEPS = 5
N_OUT_FEATS = y_train.shape[-1]
N_INP_FEATS = X_train.shape[-1]

y_train = y_train.reshape((-1, SLIDINGW_STEPS*N_OUT_FEATS))
X_train = X_train.reshape((-1, SLIDINGW_STEPS*N_INP_FEATS))

best_params = {
    'n_estimators': 100,
    'max_depth': 32,
    'max_samples': 0.75,
    'max_features': 0.75,
    'max_samples_leaf': 5,
}

qrf_best = qrf\
    .RandomForestQuantileRegressor(**best_params, n_jobs=-1, random_state=42, verbose=2)\
    .fit(X_train, y_train)


X_val = np.load('output/data_yipeng3/qrf/X_val.npy')
y_val = np.load('output/data_yipeng3/qrf/y_val.npy')

y_val = y_val.reshape((-1, SLIDINGW_STEPS*N_OUT_FEATS))
X_val = X_val.reshape((-1, SLIDINGW_STEPS*N_INP_FEATS))

y_val_pred = qrf_best.predict(X_val, quantiles=[0.025, 0.975])
y_val_low = y_val_pred[:, :, 0]
y_val_high = y_val_pred[:, :, 1]

val_mmsis = np.load('output/data_yipeng3/qrf/val-mmsis.npy', allow_pickle=True)
val_days = np.load('output/data_yipeng3/qrf/val-days.npy', allow_pickle=True)

in_pi_mask = (y_val > y_val_low) & (y_val < y_val_high)
in_pi_mask = in_pi_mask.reshape((-1, 5, 3))

group_results = {}
for ix, (mmsi, day) in enumerate(zip(val_mmsis, val_days)):

    if (mmsi, day) not in group_results:
        group_results[(mmsi, day)] = [ in_pi_mask[ix, :, :] ]
    else:
        group_results[(mmsi, day)].append( in_pi_mask[ix, :, :] )

df = []

for (mmsi, day), pi_masks in group_results.items():
    means = np.array(pi_masks).mean(axis=1).mean(axis=0)
    df.append({
        'mmsi': mmsi,
        'day': day,
        'delta_cog': means[0],
        'delta_dif': means[1],
        'dif': means[2]
    })

df = pd.DataFrame(df)

thresh_delta_cog = np.quantile(df['delta_cog'], 0.05)
thresh_delta_dif = np.quantile(df['delta_dif'], 0.05)
thresh_dif = np.quantile(df['dif'], 0.05)

print( 'delta_cog limit:', np.quantile(df['delta_cog'], 0.05) )
print( 'delta_dif liimt:', np.quantile(df['delta_dif'], 0.05) )
print( 'dif liimt:', np.quantile(df['dif'], 0.05) )

yipeng3_y = np.load('output/data_yipeng3/qrf/yipeng3_y.npy', allow_pickle=True)\
    .reshape((-1, 5*3))
yipeng3_x = np.load('output/data_yipeng3/qrf/yipeng3_x.npy', allow_pickle=True)\
    .reshape((-1, 5*20))

yipeng3_model_out = qrf_best.predict(yipeng3_x, quantiles=[0.025, 0.975])
yipeng3_low, yipeng3_high = yipeng3_model_out[:, :, 0], yipeng3_model_out[:, :, 1]

for feat in range(3):
    print(
        np.mean( (
            (yipeng3_low[75:, feat] < yipeng3_y[75:, feat]) &
            (yipeng3_high[75:, feat] > yipeng3_y[75:, feat])
        ) )
    )
