# A script to plot the count of trees in QRF against the loss
# Run after running ...
# AI was not used

import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open('../../output/models/qrf/qrf-tree-count-losses.pkl', 'rb+') as f:
    tree_counts_losses = pickle.load(f)

plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'

df_trees_loss = pd.DataFrame(tree_counts_losses).melt()\
    .rename(columns={'variable': 'N trees', 'value': 'Loss'})

sns.boxplot(df_trees_loss, x='N trees', y='Loss')
plt.ylabel('Quanile loss (\u03C4=0.025)', size=16)
plt.xlabel('Number of trees in QRF', size=16)
plt.title('QRF number of trees vs quantile loss', size=20)
plt.tight_layout()
plt.savefig('../images/tree-count-ev.png', dpi=750)

