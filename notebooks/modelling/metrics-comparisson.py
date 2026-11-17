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

from itertools import cycle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 7))
# Note: With axs = axs.T, indexing is axs[column][row]
axs = axs.T

# --- COLUMN 0: PICP (Row 0 = EV, Row 1 = YP3) ---
sns.lineplot(
    data=df[(df['Dataset'] == 'ev') & (df['Metric'] == 'picp')],
    x='time_step',
    y='dif',
    hue='Architecture',
    markers='o',
    errorbar=('pi', 95),
    style='Architecture',
    ax=axs[0][0],
)
axs[0][0].set_title('Dataset EV', size=20)
axs[0][0].set_ylabel(r'$DIF\ $ PICP', size=18)
axs[0][0].set_xticklabels(labels=['']*5)
axs[0][0].set_xlabel('')
axs[0][0].tick_params(axis='both', labelsize=18)
axs[0][0].set_yticks(ticks=[0.9, 0.925, 0.95, 0.975, 1.0], labels=[0.9, 0.925, 0.95, 0.975, 1.0])
axs[0][0].legend().remove()
axs[0][0].set_ylim((0.9, 1.0))

sns.lineplot(
    data=df[(df['Dataset'] == 'yp3') & (df['Metric'] == 'picp')],
    x='time_step',
    y='dif',
    hue='Architecture',
    markers='o',
    errorbar=('pi', 95),
    style='Architecture',
    ax=axs[0][1],
)
axs[0][1].set_title('Dataset YP3', size=20)
axs[0][1].set_ylabel(r'$DIF\ $ PICP', size=18)
axs[0][1].tick_params(axis='both', labelsize=18)
axs[0][1].set_xlabel('Time step', size=20)
axs[0][1].set_xticks(ticks=range(5, 30, 5), labels=range(5, 30, 5))
axs[0][0].set_yticks(ticks=[0.9, 0.925, 0.95, 0.975, 1.0], labels=[0.9, 0.925, 0.95, 0.975, 1.0])
axs[0][1].legend(loc='lower right', framealpha=0.7, ncols=3, fontsize=12)
axs[0][1].set_ylim((0.9, 1.0))


# --- COLUMN 1: PINAW (Row 0 = EV, Row 1 = YP3) ---
sns.lineplot(
    data=df[(df['Dataset'] == 'ev') & (df['Metric'] == 'pinaw')],
    x='time_step',
    y='dif',
    hue='Architecture',
    style='Architecture',
    markers='o',
    errorbar=('pi', 95),
    ax=axs[1][0],
)
axs[1][0].set_title('Dataset EV', size=20)
axs[1][0].set_ylabel(r'$DIF\ $ PINAW', size=18)
axs[0][0].set_xticklabels(labels=['']*5)
axs[1][0].set_xlabel('')
axs[1][0].tick_params(axis='both', labelsize=18)
axs[1][0].set_ylim((0.0, 0.65))
axs[1][0].legend().remove()

sns.lineplot(
    data=df[(df['Dataset'] == 'yp3') & (df['Metric'] == 'pinaw')],
    x='time_step',
    y='dif',
    hue='Architecture',
    style='Architecture',
    markers='o',
    errorbar=('pi', 95),
    ax=axs[1][1],
)
axs[1][1].set_title('Dataset YP3', size=20)
axs[1][1].set_xlabel('Time step', size=20)
axs[1][1].set_ylabel(r'$DIF\ $ PINAW', size=18)
axs[1][1].tick_params(axis='both', labelsize=18)
axs[1][1].set_xticks(ticks=range(5, 30, 5), labels=range(5, 30, 5))
axs[1][1].set_ylim((0.0, 0.55))
axs[1][1].legend(loc='lower right', framealpha=0.7, ncols=3, fontsize=12)


# --- COLUMN 2: BARPLOTS (Row 0 = EV, Row 1 = YP3) ---
fig.suptitle('Loss of lower and upper bound models', size=26)

hatches = ['//', '..', '++']
colors = ['#007191', '#f47a00', '#d31f11']
legend_patches = [
    mpatches.Patch(facecolor=colors[0], edgecolor='white', hatch=hatches[0], alpha=0.99, label='LSTM-AE'),
    mpatches.Patch(facecolor=colors[1], edgecolor='white', hatch=hatches[1], alpha=0.99, label='QRF'),
    mpatches.Patch(facecolor=colors[2], edgecolor='white', hatch=hatches[2], alpha=0.99, label='XGBoost'),
]

# Row 0, Col 2: Dataset EV
sns.barplot(
    data=df[(df['Dataset'] == 'ev') & (df['Metric'].isin(['loss_low', 'loss_high']))],
    x='Metric',
    y='dif',
    hue='Architecture',
    errorbar=('ci', 95),
    ax=axs[2][0],
    palette=sns.color_palette(colors),
    alpha=0.8,
)
hatches_cycle = cycle(hatches)
for i, patch in enumerate(axs[2][0].patches):
    if i % 2 == 0:
        hatch = next(hatches_cycle)
    patch.set_hatch(hatch)

axs[2][0].legend().remove()
axs[2][0].set_xticklabels(['Loss lower bound', 'Loss upper bound'], size=16)
axs[2][0].set_ylabel('Metric value', size=18)
axs[2][0].tick_params(axis='both', labelsize=14)
axs[2][0].set_xlabel('')
axs[2][0].set_title('Dataset EV', size=20)

# Row 1, Col 2: Dataset YP3
sns.barplot(
    data=df[(df['Dataset'] == 'yp3') & (df['Metric'].isin(['loss_low', 'loss_high']))],
    x='Metric',
    y='dif',
    hue='Architecture',
    errorbar=('ci', 95),
    ax=axs[2][1],
    palette=sns.color_palette(colors),
    alpha=0.8,
)
hatches_cycle = cycle(hatches)
for i, patch in enumerate(axs[2][1].patches):
    if i % 2 == 0:
        hatch = next(hatches_cycle)
    patch.set_hatch(hatch)

axs[2][1].legend(handles=legend_patches, fontsize=14, loc='lower left')
axs[2][1].set_xticklabels(['Loss lower bound', 'Loss upper bound'], size=16)
axs[2][1].set_ylabel('Metric value', size=18)
axs[2][1].tick_params(axis='both', labelsize=14)
axs[2][1].set_xlabel('Metric', size=16)
axs[2][1].set_title('Dataset YP3', size=20)

fig.tight_layout()
fig.canvas.draw()
rend = fig.canvas.get_renderer()

gap_in = 1.0  # desired gap (inches) between column 2 and column 3

# Right-most extent of column 2 (in pixels)
col2_right = max(ax.get_tightbbox(rend).x1 for ax in axs[1, :])

# Left-most extent of column 3 (in pixels)
col3_left = min(ax.get_tightbbox(rend).x0 for ax in axs[2, :])

# Shift column 3 to the right
extra = (gap_in * fig.dpi - (col3_left - col2_right)) / fig.bbox.width
for ax in axs[2, :]:
    p = ax.get_position()
    ax.set_position([p.x0 + extra, p.y0, p.width, p.height])

# Redraw so tightbboxes reflect the new positions
fig.canvas.draw()

# Center (a) under columns 0–1, and (b) under column 2
col0_left = min(ax.get_tightbbox(rend).x0 for ax in axs[0, :]) / fig.bbox.width
col1_right = max(ax.get_tightbbox(rend).x1 for ax in axs[1, :]) / fig.bbox.width
col2_left_fig = min(ax.get_tightbbox(rend).x0 for ax in axs[2, :]) / fig.bbox.width
col2_right_fig = max(ax.get_tightbbox(rend).x1 for ax in axs[2, :]) / fig.bbox.width

bottom_y = min(ax.get_tightbbox(rend).y0 for ax in axs.flat) / fig.bbox.height - 0.04

fig.text((col0_left + col1_right) / 2, bottom_y, '$(a)$', size=20, ha='center', va='top')
fig.text((col2_left_fig + col2_right_fig) / 2, bottom_y, '$(b)$', size=20, ha='center', va='top')

# Do NOT call fig.tight_layout() again here
fig.savefig(
    '../images/metrics-plot.tif',
    format='tiff',
    dpi=1000,
    bbox_inches='tight',
    pil_kwargs={'compression': 'tiff_lzw'},
)

fig.savefig('../images/metrics-plot.tif', format='tiff', dpi=1000, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
