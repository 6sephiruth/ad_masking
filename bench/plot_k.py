import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import pandas as pd
import seaborn as sns

from scipy.special import binom

# datasets
datasets = ['MNIST', 'CIFAR10', 'SVHN']
num_layers = [3, 4, 4]
colors = ['b', 'orange', 'r']

plt.figure(figsize=(6,4))
sns.set_theme(context="paper")
sns.color_palette("hls", 8)

for d,n,c in zip(datasets, num_layers, colors):
    df = pd.read_csv(f'bench_{d.lower()}.tsv', sep='\t')

    drop_cols = ['mode','model','atk_method','atk_epsilon']
    df.drop(columns=drop_cols, inplace=True)

    df['q_mask'] = df['q_mask'].apply(lambda x: round(x*n))

    sns.lineplot(x='q_mask', y='k_adv', data=df,
                 label=r'$k$-$adv$' + ' ({})'.format(d),
                 marker='o', color=c, ci=50)
    sns.lineplot(x='q_mask', y='k_over', data=df,
                 label=r'$k$-$overall$' + ' ({})'.format(d),
                 marker='X', color=c, ci=50)
    #sns.lineplot(x='mode', y='k_over', hue='attr_method', data=df)

plt.ylabel(r'optimal value of $k$')
plt.xlabel(r'number of masked layers')

plt.xticks([1,2,3,4])
plt.rcParams.update({"font.family": "serif"})
plt.legend()
plt.tight_layout()

plt.savefig('k_by_layers.png', dpi=300)