# This file is responsible for all outputs regarding validation.
# This should be run last, using the venv environment.

# Refactored with assistance from OpenAI ChatGPT (GPT-6).
# Changes made: Removed Jupyter/nbconvert cell markers and empty cells,
# converted notebook display expressions into explicit print() calls,
# and removed only redundant imports. Preserved all plotting and figure
# generation code, original calculations, input paths, and attributions.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle


plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'


metric_outputs = [
    'output/metrics/picp-lstm-ev.csv',
    'output/metrics/picp-lstm-yp3.csv',
    'output/metrics/picp-qrf-ev.csv',
    'output/metrics/picp-qrf-yp3.csv',
    'output/metrics/picp-xgb-ev.csv',
    'output/metrics/picp-xgb-yp3.csv',
    'output/metrics/pinaw-lstm-ev.csv',
    'output/metrics/pinaw-lstm-yp3.csv',
    'output/metrics/pinaw-qrf-ev.csv',
    'output/metrics/pinaw-qrf-yp3.csv',
    'output/metrics/pinaw-xgb-ev.csv',
    'output/metrics/pinaw-xgb-yp3.csv',
    'output/metrics/loss_low-lstm-ev.csv',
    'output/metrics/loss_low-lstm-yp3.csv',
    'output/metrics/loss_low-qrf-ev.csv',
    'output/metrics/loss_low-qrf-yp3.csv',
    'output/metrics/loss_low-xgb-ev.csv',
    'output/metrics/loss_low-xgb-yp3.csv',
    'output/metrics/loss_high-lstm-ev.csv',
    'output/metrics/loss_high-lstm-yp3.csv',
    'output/metrics/loss_high-qrf-ev.csv',
    'output/metrics/loss_high-qrf-yp3.csv',
    'output/metrics/loss_high-xgb-ev.csv',
    'output/metrics/loss_high-xgb-yp3.csv',
]

df = pd.DataFrame()

for metric_path in metric_outputs:
    metric, arch, dataset = tuple(metric_path.split('-'))
    metric = metric.replace('output/metrics/', '')
    dataset = dataset.split('.')[0]

    tmp_df = pd.read_csv(f'../../{metric_path}')

    if 'feature' in tmp_df:
        value_var = tmp_df.columns[1]
        tmp_df = tmp_df.pivot(
            index=['time_step', 'Fold'],
            columns='feature',
            values=value_var
        ).reset_index()

    tmp_df['Metric'] = metric
    tmp_df['Architecture'] = arch
    tmp_df['Dataset'] = dataset

    df = pd.concat([df, tmp_df], ignore_index=True)

df.drop(columns=df.columns[0], inplace=True)

print(df)


rename_arch = {
    'lstm': 'LSTM-AE',
    'qrf': 'QRF',
    'xgb': 'XGBoost'
}

df['Architecture'] = df['Architecture'].apply(
    lambda arch: rename_arch[arch]
)


fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

sns.lineplot(
    data=df[
        (df['Dataset'] == 'ev') &
        (df['Metric'] == 'picp')
    ],
    x='time_step',
    y='dif',
    hue='Architecture',
    markers='o',
    errorbar=('se', 3),
    style='Architecture',
    ax=ax1
)

sns.lineplot(
    data=df[
        (df['Dataset'] == 'ev') &
        (df['Metric'] == 'pinaw')
    ],
    x='time_step',
    y='dif',
    hue='Architecture',
    style='Architecture',
    markers='o',
    errorbar=('se', 3),
    ax=ax2
)

ax1.set_ylabel(r'$DIF\ $ PICP', size=16)
ax1.tick_params(axis='y', labelsize=14)
ax1.legend(loc='lower right', framealpha=0.7)
ax1.set_yticks(
    ticks=[0.9, 0.95, 1.0],
    labels=[0.9, 0.95, 1.0]
)
ax1.set_ylim((0.9, 1.0))

ax2.set_xlabel('Time step', size=16)
ax2.tick_params(axis='both', labelsize=14)
ax2.set_ylabel(r'$DIF\ $ PINAW', size=16)
ax2.legend(loc='lower right', framealpha=0.7)
ax2.set_xticks(
    ticks=range(5, 30, 5),
    labels=range(5, 30, 5)
)
ax2.set_ylim((0.0, 0.65))

ax1.set_title('PI metrics in EV dataset', size=20)

fig.tight_layout()
fig.savefig('../images/dataset-ev-metrics.png', dpi=750)


fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

sns.lineplot(
    data=df[
        (df['Dataset'] == 'yp3') &
        (df['Metric'] == 'picp')
    ],
    x='time_step',
    y='dif',
    hue='Architecture',
    markers='o',
    errorbar=('se', 3),
    style='Architecture',
    ax=ax1
)

sns.lineplot(
    data=df[
        (df['Dataset'] == 'yp3') &
        (df['Metric'] == 'pinaw')
    ],
    x='time_step',
    y='dif',
    hue='Architecture',
    style='Architecture',
    markers='o',
    errorbar=('se', 3),
    ax=ax2
)

ax1.set_ylabel(r'$DIF\ $ PICP', size=16)
ax1.tick_params(axis='y', labelsize=14)
ax1.legend(loc='lower right', framealpha=0.7)
ax1.set_yticks(
    ticks=[0.9, 0.95, 1.0],
    labels=[0.9, 0.95, 1.0]
)
ax1.set_ylim((0.9, 1.0))

ax2.set_xlabel('Time step', size=16)
ax2.tick_params(axis='both', labelsize=14)
ax2.set_ylabel(r'$DIF\ $ PINAW', size=16)
ax2.legend(loc='lower right', framealpha=0.7)
ax2.set_xticks(
    ticks=range(5, 30, 5),
    labels=range(5, 30, 5)
)
ax2.set_ylim((0.0, 0.8))

ax1.set_title('PI metrics in YP3 dataset', size=20)

fig.tight_layout()
fig.savefig('../images/dataset-yp3-metrics.png', dpi=750)


# Parts of code from https://github.com/mwaskom/seaborn/issues/1456#issuecomment-399002203:

fig, axs = plt.subplots(
    nrows=2,
    figsize=(6, 6),
    sharex=True
)

fig.suptitle('Loss of lower and upper bound models', size=20)

hatches = ['//', 'o', '+']
hatches_cycle = cycle(hatches)
colors = ['#85c63d', '#3d85c6', '#c63d85']

sns.barplot(
    data=df[
        (df['Dataset'] == 'yp3') &
        ((df['Metric'] == 'loss_low') | (df['Metric'] == 'loss_high'))
    ],
    x='Metric',
    y='dif',
    hue='Architecture',
    errorbar=('ci', 95),
    ax=axs[0],
    palette=sns.color_palette(colors)
)

num_locations = 2

for i, patch in enumerate(axs[0].patches):
    if i % num_locations == 0:
        hatch = next(hatches_cycle)
    patch.set_hatch(hatch)


# Source - https://stackoverflow.com/a/44937210
# Posted by Andras Deak -- Слава Україні
# Retrieved 2026-09-25, License - CC BY-SA 3.0

import matplotlib.patches as mpatches


circ1 = mpatches.Patch(
    facecolor=colors[0],
    hatch=hatches[0],
    label='LSTM-AE'
)

circ2 = mpatches.Patch(
    facecolor=colors[1],
    hatch=hatches[1],
    label='QRF'
)

circ3 = mpatches.Patch(
    facecolor=colors[2],
    hatch=hatches[2],
    label='XGBoost'
)

axs[0].legend(handles=[circ1, circ2, circ3])
plt.tight_layout()

axs[0].set_xticklabels(
    ['Loss lower bound', 'Loss upper bound'],
    size=16
)
axs[0].set_ylabel('$DIF$', size=18)
axs[0].tick_params(axis='both', labelsize=14)
axs[0].set_xlabel('')
axs[0].set_title('Dataset EV', size=15)


hatches = ['//', 'o', '+']
hatches_cycle = cycle(hatches)
colors = ['#85c63d', '#3d85c6', '#c63d85']

sns.barplot(
    data=df[
        (df['Dataset'] == 'ev') &
        ((df['Metric'] == 'loss_low') | (df['Metric'] == 'loss_high'))
    ],
    x='Metric',
    y='dif',
    hue='Architecture',
    errorbar=('ci', 95),
    ax=axs[1],
    palette=sns.color_palette(colors)
)

num_locations = 2

for i, patch in enumerate(axs[1].patches):
    if i % num_locations == 0:
        hatch = next(hatches_cycle)
    patch.set_hatch(hatch)


# Source - https://stackoverflow.com/a/44937210
# Posted by Andras Deak -- Слава Україні
# Retrieved 2026-09-25, License - CC BY-SA 3.0

circ1 = mpatches.Patch(
    facecolor=colors[0],
    hatch=hatches[0],
    label='LSTM-AE'
)

circ2 = mpatches.Patch(
    facecolor=colors[1],
    hatch=hatches[1],
    label='QRF'
)

circ3 = mpatches.Patch(
    facecolor=colors[2],
    hatch=hatches[2],
    label='XGBoost'
)

axs[1].legend(handles=[circ1, circ2, circ3])
plt.tight_layout()

axs[1].set_xticklabels(
    ['Loss lower bound', 'Loss upper bound'],
    size=16
)
axs[0].set_ylabel('Metirc value', size=18)
axs[1].set_ylabel('Metirc value', size=18)
axs[1].tick_params(axis='both', labelsize=14)
axs[1].set_xlabel('Metric', size=16)
axs[1].set_title('Dataset YP3', size=15)

fig.tight_layout()
fig.savefig('../images/loss-ev-yp3.png', dpi=750)
