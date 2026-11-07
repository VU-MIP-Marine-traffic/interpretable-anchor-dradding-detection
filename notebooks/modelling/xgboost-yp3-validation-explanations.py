"""
Responsible for XGBoost Yi-Peng 3 case validation, SHAP, MDI values

Refactored with assistance from Anthropic Claude (Claude Sonnet 5.5).
Changes made:
- Removed the shebang, coding line, Jupyter/nbconvert cell markers, and empty cells.
- Removed the unused import sklearn.metrics.mean_pinball_loss and moved the
  mid-file `import shap` to the top.
- Converted notebook cell output expressions (PI coverage means, false-positive
  rate, anomaly-free fraction, MDI tables, hard-coded MDI ratios) into
  explicit, labelled print() calls.
- Replaced plt.show() with plt.close(); the two bare exploratory line plots
  are now saved as figures instead of being displayed.
- All figures are saved to ../images/ (the MDI and SHAP figures were
  previously saved to the current working directory).
- All data inputs are read from ../../output/data_yipeng3/qrf/.
- Loaded the val/test MMSI and day arrays once instead of in several cells.
- Dropped the duplicated best_hyper definition before the Delta DIF section
  (identical to the one before Delta COG).
- Consolidated the two duplicated NMR cells into one reusable function and
  the duplicated MDI table/plot code into one helper; call parameters are
  preserved.
- Removed stale comments (commented-out debug prints in the NMR code).
- Fixed the title of the lower-bound SHAP plot, which said "upper".
- Preserved the original modeling logic and parameter choices.
"""

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


# ## Delta COG

best_hyper = {
    'n_estimators': 128,
    'max_depth': 4,
    'eta': 0.05,
    'subsample': 0.6
}

var_num = 0

model_low = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=0.025,
    device='gpu',
    **best_hyper
)

model_high = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=0.975,
    device='gpu',
    **best_hyper
)

model_low.fit(X_train, y_train[:, var_num])
model_high.fit(X_train, y_train[:, var_num])

yipeng3_low = model_low.predict(yipeng3_x)
yipeng3_high = model_high.predict(yipeng3_x)

y_low = model_low.predict(X_val)
y_high = model_high.predict(X_val)

in_pi_mask = (y_val[:, var_num] > y_low) & (y_val[:, var_num] < y_high)

group_results = {}
for ix, (mmsi, day) in enumerate(zip(val_mmsis, val_days)):
    if (mmsi, day) not in group_results:
        group_results[(mmsi, day)] = [in_pi_mask[ix]]
    else:
        group_results[(mmsi, day)].append(in_pi_mask[ix])

df = []

for (mmsi, day), pi_masks in group_results.items():
    means = np.mean(pi_masks)
    df.append({
        'mmsi': mmsi,
        'day': day,
        'delta_cog': means
    })

df = pd.DataFrame(df)
print('Delta COG:', np.quantile(df['delta_cog'], 0.05))

print(
    'YI-PENG 3 Delta COG',
    np.mean(
        (yipeng3_y[:, var_num] < yipeng3_high[:]) &
        (yipeng3_y[:, var_num] > yipeng3_low[:])
    )
)

fig, ax = plt.subplots()
ax.plot(yipeng3_y[:, var_num])
ax.plot(yipeng3_high)
ax.plot(yipeng3_low)
fig.savefig(IMAGE_DIR / 'delta-cog-xgboost-yipeng3.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)


# ## Delta DIF

var_num = 1

model_low = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=0.025,
    device='gpu',
    **best_hyper
)

model_high = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=0.975,
    device='gpu',
    **best_hyper
)

model_low.fit(X_train, y_train[:, var_num])
model_high.fit(X_train, y_train[:, var_num])

yipeng3_low = model_low.predict(yipeng3_x)
yipeng3_high = model_high.predict(yipeng3_x)

y_low = model_low.predict(X_val)
y_high = model_high.predict(X_val)

in_pi_mask = (y_val[:, var_num] > y_low) & (y_val[:, var_num] < y_high)

group_results = {}
for ix, (mmsi, day) in enumerate(zip(val_mmsis, val_days)):
    if (mmsi, day) not in group_results:
        group_results[(mmsi, day)] = [in_pi_mask[ix]]
    else:
        group_results[(mmsi, day)].append(in_pi_mask[ix])

df = []

for (mmsi, day), pi_masks in group_results.items():
    means = np.mean(pi_masks)
    df.append({
        'mmsi': mmsi,
        'day': day,
        'delta_dif': means
    })

df = pd.DataFrame(df)
print('Delta DIF:', np.quantile(df['delta_dif'], 0.05))

print(
    'YI-PENG 3 Delta DIF',
    np.mean(
        (yipeng3_y[:, var_num] < yipeng3_high[:]) &
        (yipeng3_y[:, var_num] > yipeng3_low[:])
    )
)

fig, ax = plt.subplots()
ax.plot(yipeng3_y[:, var_num])
ax.plot(yipeng3_high)
ax.plot(yipeng3_low)
fig.savefig(IMAGE_DIR / 'delta-dif-xgboost-yipeng3.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)


# ## DIF

best_hyper = {
    'n_estimators': 128,
    'max_depth': 4,
    'eta': 0.05,
    'subsample': 1.0
}

var_num = 2

model_low = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=0.025,
    device='gpu',
    **best_hyper
)

model_high = xgb.XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=0.975,
    device='gpu',
    **best_hyper
)

model_low.fit(X_train, y_train[:, var_num])
model_high.fit(X_train, y_train[:, var_num])

yipeng3_low = model_low.predict(yipeng3_x)
yipeng3_high = model_high.predict(yipeng3_x)

y_low = model_low.predict(X_val)
y_high = model_high.predict(X_val)

in_pi_mask = (y_val[:, var_num] > y_low) & (y_val[:, var_num] < y_high)

group_results = {}
for ix, (mmsi, day) in enumerate(zip(val_mmsis, val_days)):
    if (mmsi, day) not in group_results:
        group_results[(mmsi, day)] = [in_pi_mask[ix]]
    else:
        group_results[(mmsi, day)].append(in_pi_mask[ix])

df = []

for (mmsi, day), pi_masks in group_results.items():
    means = np.mean(pi_masks)
    df.append({
        'mmsi': mmsi,
        'day': day,
        'delta_cog': means
    })

df = pd.DataFrame(df)
thresh_dif = np.quantile(df['delta_cog'], 0.05)
print('DIF:', thresh_dif)

print(
    'YI-PENG 3 DIF (first 75 steps)',
    np.mean(
        (yipeng3_y[:75, var_num] < yipeng3_high[:75]) &
        (yipeng3_y[:75, var_num] > yipeng3_low[:75])
    )
)


# False positive rate on the test set (DIF)
y_low = model_low.predict(X_test)
y_high = model_high.predict(X_test)

in_pi_mask = (y_test[:, var_num] > y_low) & (y_test[:, var_num] < y_high)

group_results = {}
for ix, (mmsi, day) in enumerate(zip(test_mmsis, test_days)):
    if (mmsi, day) not in group_results:
        group_results[(mmsi, day)] = [in_pi_mask[ix]]
    else:
        group_results[(mmsi, day)].append(in_pi_mask[ix])

df = []

for (mmsi, day), pi_masks in group_results.items():
    means = np.mean(pi_masks)
    df.append({
        'mmsi': mmsi,
        'day': day,
        'dif': means
    })

df = pd.DataFrame(df)
print('FPR for DIF is', np.mean(df['dif'] < thresh_dif))


timestamps_yipeng3 = pd.date_range(
    start=datetime(2024, 11, 18, 2, 30),
    end=datetime(2024, 11, 19, 0, 0),
    periods=yipeng3_y.shape[0]
)

y_flat = yipeng3_y[:75, 2]
y_low = yipeng3_low[:75]
y_high = yipeng3_high[:75]
timestamps_yipeng3 = list(timestamps_yipeng3)[:75]

anomalies_x = []
anomalies_y = []
for i in range(y_flat.shape[0]):
    if y_flat[i] < y_low[i]:
        anomalies_x.append(timestamps_yipeng3[i])
        anomalies_y.append(y_flat[i])

    elif y_flat[i] > y_high[i]:
        anomalies_x.append(timestamps_yipeng3[i])
        anomalies_y.append(y_flat[i])

fig = plt.figure(figsize=(10,6))

plt.plot(
    timestamps_yipeng3,
    y_flat,
    label="Target value",
    color='black',
    alpha=0.6
)
plt.plot(timestamps_yipeng3, y_low, color='#185288', linestyle='--')
plt.plot(timestamps_yipeng3, y_high, color="#185288", linestyle='--')

ticks = [
    datetime(2024, 11, 18, 0, 0) + timedelta(hours=x)
    for x in range(0, 30, 3)
]
labels = [tick.strftime('%Y-%m-%d\n%H:%M') for tick in ticks]
plt.xticks(
    ticks=ticks,
    labels=labels
)
plt.tick_params(axis='both', which='both', labelsize=20)

plt.xlabel('Timestamp (UTC)', size=24)
plt.ylabel(r'$DIF$', size=24)
plt.title(r'Yi-Peng 3 $DIF$ PI', size=32)

plt.fill_between(
    x=timestamps_yipeng3,
    y1=y_low,
    y2=y_high,
    alpha=0.3,
    label='Prediction interval'
)
plt.scatter(
    anomalies_x,
    anomalies_y,
    color='red',
    s=20,
    zorder=5,
    label='Anomalies'
)
plt.legend(fontsize=20)

plt.tight_layout()
plt.savefig(IMAGE_DIR / 'dif-xgboost-yipeng3.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

print(
    'YI-PENG 3 DIF fraction of points inside PI (first 75 steps):',
    (75 - len(anomalies_y)) / 75
)


features = [
    'SPEED', 'COURSE', 'currentSpeed', 'gust', 'swellHeight',
    'waveHeight', 'windSpeed', 'dif', 'COURSE-dif', 'delta_dif',
    'COURSE_vec_x', 'COURSE_vec_y', 'currentDirection_vec_x',
    'currentDirection_vec_y', 'swellDirection_vec_x',
    'swellDirection_vec_y', 'waveDirection_vec_x', 'waveDirection_vec_y',
    'windDirection_vec_x', 'windDirection_vec_y'
]
feature_names = [
    'SOG', 'COG', 'CurrentSpeed', 'Gust', 'SwellHeight',
    'WaveHeight', 'WindSpeed', 'DIF', r'\Delta COG',
    r'\Delta DIF', 'COG_x', 'COG_y', 'CurrentDirection_x',
    'CurrentDirection_y', 'SwellDirection_x', 'SwellDirection_y',
    'WaveDirection_x', 'WaveDirection_y', ' WindDirection_x', 'WindDirection_y'
]

feature_names = [
    rf'${col}(t-{t})$'
    for t in range(0, 25, 5)
    for col in feature_names
]

explainer_high = shap.TreeExplainer(model_high, feature_names=feature_names)
explainer_low = shap.TreeExplainer(model_low, feature_names=feature_names)

shap_high = explainer_high(yipeng3_x)
shap_low = explainer_low(yipeng3_x)


def top_mdi(model, n=10):
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Feature importance': model.feature_importances_
    })
    return (
        df_imp
        .set_index(keys='Feature')
        .sort_values(by='Feature importance', ascending=True)
        .tail(n)
    )


fig, axs = plt.subplots(nrows=2, figsize=(6, 6))
fig.suptitle('MDI scores of XGBoost models\nDataset YP3, $DIF$ target', size=20)

shap_top_attrib_low = top_mdi(model_low)
shap_top_attrib_low.plot.barh(ax=axs[0])

# Source - https://stackoverflow.com/a/26845924
# Posted by naitsirhc, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-23, License - CC BY-SA 4.0
axs[0].get_legend().remove()
axs[0].tick_params(axis='both', labelsize=14)
axs[0].set_ylabel('Feature', size=18)
axs[0].set_title('Lower bound', size=20)

top_mdi(model_high).plot.barh(ax=axs[1])

# Source - https://stackoverflow.com/a/26845924
# Posted by naitsirhc, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-23, License - CC BY-SA 4.0
axs[1].get_legend().remove()
axs[1].tick_params(axis='both', labelsize=14)
axs[1].set_ylabel('Feature', size=18)
axs[1].set_xlabel('MDI', size=18)
axs[1].set_title('Upper bound', size=20)

plt.tight_layout()
plt.savefig(IMAGE_DIR / 'mdi-scores-xgboost-yp3.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)

print('MDI scores, lower bound (top 10):')
print(top_mdi(model_low))
print('MDI ratio (lower bound):', 0.157783 / 0.038165)

print('MDI scores, upper bound (top 10):')
print(top_mdi(model_high))
print('MDI ratio (upper bound):', 0.178426 / 0.032979)


fig, axs = plt.subplots(2, 1)
plt.sca(axs[0])

shap_low_values = np.array(shap_low.values)
shap_low_base=  shap_low.base_values[0]

print( 'SHAP base value',  shap_low_base )
print( 'SHAP SOG min:', shap_low_values[:, 0].min() )

shap.plots.beeswarm(shap_low[:75], show=False, plot_size=(8, 8), max_display=11)

axs[0].set_title(
    'SHAP values for XGBoost (lower) \nfor the Yi-Peng 3 case, $DIF$ target',
    size=20
)
axs[0].tick_params(axis='both', labelsize=16)
axs[0].set_xlabel('SHAP value', size=18)

# https://github.com/shap/shap/issues/594#issuecomment-1206481300
cbar = plt.gcf().axes[1]
cbar.tick_params(labelsize=15)
cbar.set_ylabel("Feature value", fontsize=18)

plt.sca(axs[1])
shap.plots.beeswarm(shap_high[:75], show=False, plot_size=(8, 8), max_display=11)

axs[1].set_title(
    'SHAP values for XGBoost (upper) \nfor the Yi-Peng 3 case, $DIF$ target',
    size=20
)
axs[1].tick_params(axis='both', labelsize=16)
axs[1].set_xlabel('SHAP value', size=18)

# https://github.com/shap/shap/issues/594#issuecomment-1206481300
cbar = plt.gcf().axes[1]
cbar.tick_params(labelsize=15)
cbar.set_ylabel("Feature value", fontsize=18)

plt.tight_layout()
plt.savefig(IMAGE_DIR / 'shap-values-yipeng3.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)


# This code is adapted from https://github.com/amaa11/NMR
def add_item_to_dict(my_dict, key, value):
    my_dict[len(my_dict)] = [key, value]


def calculate_nmr(X_test_values, feature_names, quantile_alpha):
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_values, columns=feature_names)
    y_train_dif = y_train[:, 2]

    dicts = dict()
    m = 0
    MPM_list = []

    while X_train_df.shape[1] != 1:
        MPM = 0
        NF = X_train_df.shape[1]

        for j in range(1, NF, 2):
            summ = NF - j
            MPM = MPM + summ

        MPM = MPM * 2
        MPM_list.append(MPM)

        ITER_NUMBER = X_train_df.shape[1]

        model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=quantile_alpha,
            device='gpu',
            **best_hyper
        )
        model.fit(X_train_df, y_train_dif)
        ypred_test = model.predict(X_test_df)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_df)

        vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(
            list(zip(X_train_df.columns, vals)),
            columns=['Features', 'feature_importance_vals']
        )
        feature_importance.sort_values(
            by=['feature_importance_vals'],
            ascending=False,
            inplace=True
        )
        feature_importance.reset_index(inplace=True, drop=True)
        feature_importance.index = np.arange(1, len(feature_importance) + 1)

        FEATURES = feature_importance[['Features']]

        for i in range(FEATURES.shape[0]):
            if FEATURES.iloc[i]['Features'] in dicts:
                dicts[FEATURES.iloc[i]['Features']].append(
                    FEATURES.iloc[i].name + m
                )
            else:
                dicts[FEATURES.iloc[i]['Features']] = [
                    FEATURES.iloc[i].name + m
                ]

        m = m + 1
        Top_feature = feature_importance.iloc[0]['Features']

        X_train_df.drop([Top_feature], axis=1, inplace=True)
        X_train_df.reset_index(inplace=True, drop=True)
        X_test_df.drop([Top_feature], axis=1, inplace=True)
        X_test_df.reset_index(inplace=True, drop=True)

    MPM_list.pop(0)

    lenOfLists = len(dicts)

    sorted_dict = {}

    for index in range(1, lenOfLists):
        for key, value in dicts.items():
            if len(value) == index:
                add_item_to_dict(sorted_dict, key, value)

    sumOfChanges = {}
    add_item_to_dict(sumOfChanges, sorted_dict[0][0], 0)

    for index in range(1, lenOfLists - 1):
        key = sorted_dict[index][0]
        soc = 0

        for t in range(index, lenOfLists):
            soc += abs(
                sorted_dict[t][1][index] -
                sorted_dict[t][1][index - 1]
            )

        add_item_to_dict(sumOfChanges, key, soc)

    add_item_to_dict(
        sumOfChanges,
        sorted_dict[lenOfLists - 1][0],
        sumOfChanges[lenOfLists - 2][1]
    )

    sumOfChanges.popitem()
    first_key = next(iter(sumOfChanges))
    del sumOfChanges[first_key]

    rates = []
    ite = 0

    for key, value in sumOfChanges.items():
        rates.append(value[1] / MPM_list[ite])
        ite = ite + 1

    print("NMR value : ", round(np.mean(rates), 3))


print('calculating NMR for Yi-Peng 3 upper bound')
calculate_nmr(yipeng3_x, feature_names, 0.975)
print('calculating NMR for Yi-Peng 3 upper bound (repeat of the second notebook cell)')
calculate_nmr(yipeng3_x, feature_names, 0.975)
