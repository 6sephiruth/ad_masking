import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['text.usetex'] = True

# datasets
datasets = ['MNIST', 'CIFAR10', 'SVHN']
num_layers = [3, 4, 4]
colors = ['blue', 'orange', 'r']

plt.figure(figsize=(6,4))
sns.set_theme(context="paper")
sns.color_palette("hls", 8)

for d,n,c in zip(datasets, num_layers, colors):
    df = pd.read_csv(f'bench_{d.lower()}.tsv', sep='\t')

    adv_init = df['adv_init'][0]

    df['adv_over'] = (df['adv_over'] - adv_init)/100
    df['adv_adv'] = (df['adv_adv'] - adv_init)/100

    df['q_mask'] = df['q_mask'].apply(lambda x: round(x*n))

    best = [d['adv_over'].max() for v,d in df.groupby('q_mask')]
    adv = [d['adv_adv'].max() for v,d in df.groupby('q_mask')]

    #df['q_mask'] = df['q_mask'].apply(lambda x: round(x*n))

    #df['best_over'] = (df['test_over'] + df['adv_over'])/2

    sns.lineplot(x='q_mask', y='adv_adv', data=df,
                 label=r'$\mathtt{best}$-$\mathtt{adv}$' + ' ({})'.format(d),
                 marker='o', color=c, ci=50)
    sns.lineplot(x='q_mask', y='adv_over', data=df,
                 label=r'$\mathtt{best}$-$\mathtt{overall}$' + ' ({})'.format(d),
                 marker='X', color=c, ci=50)

    #attrs = set(df['attr_method'])

    #drop_cols = ['mode','model','atk_method','atk_epsilon']
    #df.drop(columns=drop_cols, inplace=True)

    #df['q_mask'] = df['q_mask'].apply(lambda x: round(x*n))

    #sns.lineplot(x='q_mask', y='k_adv', data=df,
    #             label=r'$k$-$adv$' + ' ({})'.format(d),
    #             marker='o', color=c, ci=50)
    #sns.lineplot(x='q_mask', y='k_over', data=df,
    #             label=r'$k$-$overall$' + ' ({})'.format(d),
    #             marker='X', color=c, ci=50)
    #sns.lineplot(x='mode', y='k_over', hue='attr_method', data=df)

plt.ylabel(r'adversarial accuracy gain')
plt.xlabel(r'number of masked layers')

plt.xticks([1,2,3,4])
plt.rcParams.update({"font.family": "serif"})
plt.legend()
plt.tight_layout()

plt.savefig('acc_by_layers.png', dpi=300)