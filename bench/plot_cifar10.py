import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import pandas as pd
import seaborn as sns

from scipy.special import binom

df = pd.read_csv('bench_best_by.tsv', sep='\t')

drop_cols = ['mode','data','model','atk_method','atk_epsilon']
df.drop(columns=drop_cols, inplace=True)

sns.lineplot(x='q_mask', y='k_adv', data=df, label=r'k-adv')
sns.lineplot(x='q_mask', y='k_over', data=df, label=r'k-over')
#sns.lineplot(x='mode', y='k_over', hue='attr_method', data=df)

plt.ylabel('Accuracy')
plt.xlabel(r'$k$')
plt.legend()
plt.savefig('cifar10.png')