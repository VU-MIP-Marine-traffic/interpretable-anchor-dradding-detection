import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'

y_train_yp3 = np.load('../../output/data_eagles_vezhen/qrf/y_train2.npy')
y_test_yp3 = np.load('../../output/data_eagles_vezhen/qrf/y_test2.npy')
y_val_yp3 = np.load('../../output/data_eagles_vezhen/qrf/y_val2.npy')

DATA_DIR = Path("../../output/data_yipeng3/qrf")

y_yp3 = np.concatenate([y_train_yp3, y_test_yp3, y_val_yp3])

df_yp3 = pd.DataFrame({
    r'$\Delta COG$': y_yp3[:, 0, 0],
    r'$\Delta DIF$': y_yp3[:, 0, 1],
    r'$DIF$': y_yp3[:, 0, 2],
})
df_yp3 = df_yp3.melt(
    value_vars=[r'$\Delta COG$', r'$\Delta DIF$', r'$DIF$'],
    var_name='Target variable',
    value_name='Value'
)

figure(figsize=(8, 4), dpi=1000)
sns.boxplot(
    data = df_yp3,
    y='Target variable',
    x='Value',
    orient='h',
    boxprops={'facecolor': (0.0, 0.0, 0.0, 0.0)}
)
sns.violinplot(
    data = df_yp3,
    y='Target variable',
    x='Value',
    orient='h',
    color='blue',
    alpha=0.3
)

plt.title('Distribution of target variables in YP3', size=20)
plt.tick_params(axis='both', labelsize=16)
plt.ylabel('Target variable', size=18)
plt.xlabel('Value', size=18)

plt.tight_layout()
plt.savefig('../images/Figure_A1.tif', format='tiff', dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
plt.show()

del df_yp3, y_yp3, y_train_yp3, y_test_yp3, y_val_yp3

y_train_ev = np.load(DATA_DIR / 'y_train.npy')
y_test_ev= np.load(DATA_DIR / 'y_test.npy')
y_val_ev = np.load(DATA_DIR / 'y_val.npy')
y_ev = np.concatenate([y_train_ev, y_test_ev, y_val_ev])

df_ev = pd.DataFrame({
    r'$\Delta COG$': y_ev[:, 0, 0],
    r'$\Delta DIF$': y_ev[:, 0, 1],
    r'$DIF$': y_ev[:, 0, 2],
})
df_ev = df_ev.melt(
    value_vars=[r'$\Delta COG$', r'$\Delta DIF$', r'$DIF$'],
    var_name='Target variable',
    value_name='Value'
)


figure(figsize=(8, 4), dpi=1000)
sns.boxplot(
    data = df_ev,
    y='Target variable',
    x='Value',
    orient='h',
    boxprops={'facecolor': (0.0, 0.0, 0.0, 0.0)}
)
sns.violinplot(
    data = df_ev,
    y='Target variable',
    x='Value',
    orient='h',
    color='blue',
    alpha=0.3
)

plt.title('Distribution of target variables in EV', size=20)
plt.tick_params(axis='both', labelsize=16)
plt.ylabel('Target variable', size=18)
plt.xlabel('Value', size=18)

plt.tight_layout()
plt.savefig('../images/Figure_A2.tif', format='tiff', dpi=1000, pil_kwargs={"compression": "tiff_lzw"})

del df_ev, y_ev, y_train_ev, y_test_ev, y_val_ev
