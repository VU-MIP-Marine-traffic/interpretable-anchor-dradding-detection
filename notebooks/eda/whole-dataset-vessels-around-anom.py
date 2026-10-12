# To run:

# Cleanup by ChatGPT Luna (5.6):
# - Removed all exploratory notebook cells and nbconvert cell markers.
# - Removed unrelated AIS, navigation-status, weather, and vessel-specific analysis.
# - Kept only imports required for the final three KDE figures.
# - Preserved the vessel date/time processing and course/heading calculations
#   required to produce the plotted variables.
# - Updated all three output paths to save figures under ../images/.
# - Added plt.close() after each figure to avoid accumulating open figures.
# - Added attribution to OpenAI ChatGPT (GPT-5.6 Luna).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../../data/Vessels_PositionReport_5min_25Dec2024_25Jan2025.csv')

df['DATE TIME (UTC)'] = pd.to_datetime(df['DATE TIME (UTC)'])
df['day'] = df['DATE TIME (UTC)'].dt.date


def add_course_delta(group):
    group['cog_delta_abs'] = np.abs(
        (group['COURSE'].diff() + 180) % 360 - 180
    )
    group['dif'] = (
        (group['COURSE'] - group['HEADING'] + 180) % 360 - 180
    )
    group['delta_dif_abs'] = group['dif'].diff().abs()
    group['dif_abs'] = group['dif'].abs()
    return group.drop(columns=['MMSI', 'day'])


df_ = (
    df.sort_values('DATE TIME (UTC)')
    .groupby(['MMSI', 'day'])
    .apply(add_course_delta)
    .reset_index()
    .drop(columns='level_2')
)

plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = 'white'

sns.kdeplot(data=df_, x='SPEED', y='cog_delta_abs', fill=True)
plt.ylabel(r'$|\Delta COG|$', size=16)
plt.xlabel('SOG', size=16)
plt.title(r'Absolute value of $\Delta COG$ and SOG', size=20)
plt.tight_layout()
plt.savefig('../images/delta_cog_speed.png')
plt.close()

sns.kdeplot(data=df_, x='SPEED', y='dif_abs', fill=True)
plt.ylabel(r'$|DIF|$', size=16)
plt.xlabel('SOG', size=16)
plt.title(r'Absolute value of $DIF$ and SOG', size=20)
plt.tight_layout()
plt.savefig('../images/dif_speed.png')
plt.close()

sns.kdeplot(data=df_, x='SPEED', y='delta_dif_abs', fill=True)
plt.ylabel(r'$|\Delta DIF|$', size=16)
plt.xlabel('SOG', size=16)
plt.title(r'Absolute value of $\Delta DIF$ and SOG', size=20)
plt.tight_layout()
plt.savefig('../images/delta_dif_speed.png')
plt.close()