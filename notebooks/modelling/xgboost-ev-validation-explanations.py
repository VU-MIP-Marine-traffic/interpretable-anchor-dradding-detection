# Responsible for XGBoost Eagle s and vezhen case validation, SHAP, Gain metric values

# Refactored with assistance from OpenAI ChatGPT (GPT-6).
# Changes made:
# - Removed Jupyter/nbconvert cell markers and empty cells.
# - Removed plotting and visualization code to avoid displaying or saving images.
# - Converted meaningful notebook output expressions into explicit print() calls.
# - Removed imports used only by deleted plotting code or otherwise unused.
# - Consolidated repeated NMR calculations into a reusable function while preserving the original calls and parameters.
# - Preserved the original modeling logic, including existing parameter choices and the feat = 0 assignment.
# - Preserved attributions to other sources, including the NMR GitHub repository, Stack Overflow, and the SHAP issue.
# - Removed the stale exploratory section that referenced undefined variables anomalies and df.
# - Kept data input paths under ../../output/.
# - Did not execute the script or verify runtime behavior.

import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pytz

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


FEAT_NAMES = ['Delta COG', 'Delta DIF', 'DIF']


def test_xgboost_feature(feat: int, feat_name: str, hyper: dict):
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

    return (
        eagles_y_tmp,
        eagles_low,
        eagles_high,
        vezhen_y_tmp,
        vezhen_low,
        vezhen_high
    )


best_hyper = {
    'n_estimators': 128,
    'max_depth': 4,
    'eta': 0.05,
    'subsample': 0.8,
}

(
    eagles_true_delta_cog,
    eagles_low_delta_cog,
    eagles_high_delta_cog,
    vezhen_true_delta_cog,
    vezhen_low_delta_cog,
    vezhen_high_delta_cog
) = test_xgboost_feature(
    0,
    feat_name='Delta COG',
    hyper=best_hyper
)


best_hyper = {
    'n_estimators': 128,
    'max_depth': 4,
    'eta': 0.05,
    'subsample': 0.8,
}

(
    eagles_true_delta_dif,
    eagles_low_delta_dif,
    eagles_high_delta_dif,
    vezhen_true_delta_dif,
    vezhen_low_delta_dif,
    vezhen_high_delta_dif
) = test_xgboost_feature(
    1,
    feat_name='Delta DIF',
    hyper=best_hyper
)


best_hyper = {
    'n_estimators': 128,
    'max_depth': 4,
    'eta': 0.05,
    'subsample': 1.0,
}

(
    eagles_true_dif,
    eagles_low_dif,
    eagles_high_dif,
    vezhen_true_dif,
    vezhen_low_dif,
    vezhen_high_dif
) = test_xgboost_feature(
    2,
    feat_name='DIF',
    hyper=best_hyper
)


timestamps_eagles = pd.date_range(
    start=datetime(2024, 12, 25, 10, 20),
    end=datetime(2024, 12, 25, 16, 40),
    periods=eagles_x.shape[0]
)

def plot_pi(
    ship_name: str,
    feat_name: str,
    y_true: np.ndarray,
    y_low: np.ndarray,
    y_high: np.ndarray,
    plot_ticks: list,
    plot_labels: list,
    ax: plt.Axes,
    include_xtitle: bool = False,
    include_legend: bool = False,
    legend_loc: str = None,
    legend_ncols: int = 1
):
    anomalies_x = []
    anomalies_y = []

    for i in range(y_true.shape[0]):
        if y_true[i] < y_low[i]:
            anomalies_x.append(timestamps_eagles[i])
            anomalies_y.append(y_true[i])
        elif y_true[i] > y_high[i]:
            anomalies_x.append(timestamps_eagles[i])
            anomalies_y.append(y_true[i])

    ax.plot(
        timestamps_eagles,
        y_true,
        label="Target\nvalue",
        color='black',
        alpha=0.6
    )
    ax.plot(
        timestamps_eagles,
        y_low,
        color='#185288',
        linestyle='--'
    )
    ax.plot(
        timestamps_eagles,
        y_high,
        color="#185288",
        linestyle='--'
    )

    ax.axvline(
        x=datetime(2024, 12, 25, 10, 26, tzinfo=pytz.utc),
        color="#a88080",
        linestyle='--',
        label='Estlink-2 cut',
    )

    ax.set_xticks(ticks=plot_ticks, labels=plot_labels)
    ax.tick_params(axis='both', which='both', labelsize=13)

    if include_xtitle:
        ax.set_xlabel('Timestamp (UTC)', size=18)

    ax.set_ylabel(rf'{feat_name}', size=18)
    ax.set_title(rf'{ship_name} {feat_name} PI', size=20)

    ax.fill_between(
        x=timestamps_eagles,
        y1=y_low,
        y2=y_high,
        alpha=0.3,
        label='Prediction\ninterval'
    )
    ax.scatter(
        anomalies_x,
        anomalies_y,
        color='red',
        s=20,
        zorder=5,
        label='Anomalies'
    )

    if not include_legend:
        ax.legend().remove()
    elif legend_loc:
        ax.legend(fontsize=13, framealpha=0.5, loc='upper center', ncols=legend_ncols, bbox_to_anchor=(0.5, -0.4))
    else:
        ax.legend(fontsize=13, framealpha=0.5, loc='upper right', ncols=legend_ncols)


plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'

fig, axs = plt.subplots(
    nrows=3,
    figsize=(8, 8),
    gridspec_kw={'hspace': 0.25},
    sharex=True
)

ticks = [
    datetime(2024, 12, 25, 12, 0) + timedelta(hours=1) * i
    for i in range(0, 6, 2)
]

labels = [
    tick.strftime('%Y-%m-%d\n%H:%M')
    for tick in ticks
]

plot_pi(
    'EAGLE S',
    '$DIF$',
    eagles_true_dif[:, 0],
    eagles_low_dif[:, 0],
    eagles_high_dif[:, 0],
    ticks,
    labels,
    ax=axs[0],
    legend_loc='lower right',
    include_legend=False
)

plot_pi(
    'EAGLE S',
    r'$\Delta DIF$',
    eagles_true_delta_dif[:, 0],
    eagles_low_delta_dif[:, 0],
    eagles_high_delta_dif[:, 0],
    ticks,
    labels,
    ax=axs[1],
    include_legend=False
)

plot_pi(
    'EAGLE S',
    r'$\Delta COG$',
    eagles_true_delta_cog[:, 0],
    eagles_low_delta_cog[:, 0],
    eagles_high_delta_cog[:, 0],
    ticks,
    labels,
    ax=axs[2],
    include_xtitle=True,
    include_legend=True,
    legend_ncols=4,
    legend_loc='lower right'
)

plt.tight_layout()
plt.savefig(IMAGE_DIR / 'eagles-results.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)


hyper = {
    'n_estimators': 128,
    'max_depth': 4,
    'eta': 0.05,
    'subsample': 1.0,
}

feat = 2  # dif

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

y_train_tmp = y_train.reshape((-1, 5, 3))[:, 0, feat]

model_low.fit(X_train, y_train_tmp)
model_high.fit(X_train, y_train_tmp)

eagles_low_slice = model_low.predict(eagles_x)
eagles_high_slice = model_high.predict(eagles_x)


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
    rf'${col}(t-{t})$' if t != 0 else f'${col}(t)$'
    for t in range(0, 25, 5)
    for col in feature_names
]

explainer_high = shap.TreeExplainer(
    model_high,
    feature_names=feature_names
)
explainer_low = shap.TreeExplainer(
    model_low,
    feature_names=feature_names
)

shap_high = explainer_high(eagles_x)
shap_low = explainer_low(eagles_x)


fig, axs = plt.subplots(2, 1)

plt.sca(axs[0])
shap.plots.beeswarm(
    shap_low,
    show=False,
    plot_size=(8, 8),
    max_display=11
)

axs[0].set_title(
    'Lower bound model',
    size=20
)
axs[0].tick_params(axis='both', labelsize=16)
axs[0].set_xlabel('SHAP value', size=18)

# https://github.com/shap/shap/issues/594#issuecomment-1206481300
cbar = plt.gcf().axes[1]
cbar.tick_params(labelsize=15)

plt.sca(axs[1])
shap.plots.beeswarm(
    shap_high,
    show=False,
    plot_size=(8, 8),
    max_display=11
)

axs[1].set_title(
    'Upper bound model',
    size=20
)
axs[1].tick_params(axis='both', labelsize=16)
axs[1].set_xlabel('SHAP value', size=18)

# https://github.com/shap/shap/issues/594#issuecomment-1206481300
cbar = plt.gcf().axes[1]
cbar.tick_params(labelsize=15)

fig.suptitle('SHAP values for XGBoost\nEAGLE S case evaluation, $DIF\ $ target', size=26)

plt.tight_layout()
plt.savefig(IMAGE_DIR / 'shap-values-eagles.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)


shap_high = explainer_high(vezhen_x)
shap_low = explainer_low(vezhen_x)

fig, axs = plt.subplots(2, 1)

plt.sca(axs[0])
shap.plots.beeswarm(
    shap_low,
    show=False,
    plot_size=(8, 8),
    max_display=11
)

axs[0].set_title(
    'Lower bound model',
    size=20
)
axs[0].tick_params(axis='both', labelsize=16)
axs[0].set_xlabel('SHAP value', size=18)

# https://github.com/shap/shap/issues/594#issuecomment-1206481300
cbar = plt.gcf().axes[1]
cbar.tick_params(labelsize=15)

plt.sca(axs[1])
shap.plots.beeswarm(
    shap_high,
    show=False,
    plot_size=(8, 8),
    max_display=11
)

axs[1].set_title(
    'Upper bound model',
    size=20
)
axs[1].tick_params(axis='both', labelsize=16)
axs[1].set_xlabel('SHAP value', size=18)

# https://github.com/shap/shap/issues/594#issuecomment-1206481300
cbar = plt.gcf().axes[1]
cbar.tick_params(labelsize=15)

fig.suptitle('SHAP values for XGBoost\nVEZHEN case evaluation, $DIF\ $ target', size=26)

plt.tight_layout()
plt.savefig(IMAGE_DIR / 'shap-values-vezhen.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)


fig, axs = plt.subplots(nrows=2, figsize=(6, 6))
fig.suptitle(
    'Gain metric values of XGBoost models\nDataset EV, $DIF$ target',
    size=20
)

df_imp = pd.DataFrame({
    'Feature': feature_names,
    'Feature importance': model_low.feature_importances_
})

df_imp.set_index(keys='Feature') \
    .sort_values(by='Feature importance', ascending=True) \
    .tail(10) \
    .plot.barh(ax=axs[0])

# Source - https://stackoverflow.com/a/26845924
# Posted by naitsirhc, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-23, License - CC BY-SA 4.0
axs[0].get_legend().remove()
axs[0].tick_params(axis='both', labelsize=14)
axs[0].set_ylabel('Feature', size=18)
axs[0].set_title('Lower bound', size=20)

df_imp = pd.DataFrame({
    'Feature': feature_names,
    'Feature importance': model_high.feature_importances_
})

df_imp.set_index(keys='Feature') \
    .sort_values(by='Feature importance', ascending=True) \
    .tail(10) \
    .plot.barh(ax=axs[1])

# Source - https://stackoverflow.com/a/26845924
# Posted by naitsirhc, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-23, License - CC BY-SA 4.0
axs[1].get_legend().remove()
axs[1].tick_params(axis='both', labelsize=14)
axs[1].set_ylabel('Feature', size=18)
axs[1].set_xlabel('Gain', size=18)
axs[1].set_title('Upper bound', size=20)

plt.tight_layout()
plt.savefig(IMAGE_DIR / 'mdi-scores-xgboost-ev.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)


test_mmsis = np.load(
    '../../output/data_eagles_vezhen/qrf/test-mmsis2.npy',
    allow_pickle=True
)
test_days = np.load(
    '../../output/data_eagles_vezhen/qrf/test-days2.npy',
    allow_pickle=True
)


def calc_false_positve(var_num: int, thresh: float):
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

    y_low = model_low.predict(X_test)
    y_high = model_high.predict(X_test)

    in_pi_mask = (
        (y_test[:, var_num] > y_low) &
        (y_test[:, var_num] < y_high)
    )

    group_results = {}

    for ix, (mmsi, day) in enumerate(zip(test_mmsis, test_days)):
        if (mmsi, day) not in group_results:
            group_results[(mmsi, day)] = [in_pi_mask[ix]]
        else:
            group_results[(mmsi, day)].append(in_pi_mask[ix])

    df = []

    for (mmsi, day), pi_masks in group_results.items():
        mean = np.mean(pi_masks)
        df.append({
            'mmsi': mmsi,
            'day': day,
            'coverage': mean
        })

    df = pd.DataFrame(df)
    return np.mean(df['coverage'] < thresh)


# taken from the top
# TODO: save it at the top and re-use it here
thresholds = [
    0.8867307692307692,
    0.8856525423728814,
    0.8775641025641026
]

feats=['Delta COG', 'Delta DIF', 'DIF']
for i in range(3):
    print('FPR for', feats[i], 'is', calc_false_positve(i, thresholds[i]))


# from https://github.com/amaa11/NMR
def calculate_nmr(X_test_values, feature_names, quantile_alpha):
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_values, columns=feature_names)
    y_train_dif = y_train[:, 2]

    dicts = dict()
    m = 0
    MPM_list = []

    model = xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=quantile_alpha,
        device='gpu',
        **best_hyper
    )

    while X_train_df.shape[1] != 1:
        MPM = 0
        NF = X_train_df.shape[1]

        for j in range(1, NF, 2):
            summ = NF - j
            MPM = MPM + summ

        MPM = MPM * 2
        MPM_list.append(MPM)

        ITER_NUMBER = X_train_df.shape[1]

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
        feature_importance.index = np.arange(
            1, len(feature_importance) + 1
        )

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

    def add_item_to_dict(my_dict, key, value):
        my_dict[len(my_dict)] = [key, value]

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


# from https://github.com/amaa11/NMR
feature_names_latex = feature_names.copy()
y_train_dif = y_train[:, 2]

print('calculating NMR for EAGLE S lower bound')
calculate_nmr(eagles_x, feature_names_latex, 0.025)
print('calculating NMR for EAGLE S upper bound')
calculate_nmr(eagles_x, feature_names_latex, 0.975)
print('calculating NMR for Yi-Peng 3 lower bound')
calculate_nmr(vezhen_x, feature_names_latex, 0.025)
print('calculating NMR for Yi-Peng 3 upper bound')
calculate_nmr(vezhen_x, feature_names_latex, 0.975)
