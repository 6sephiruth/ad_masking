import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['text.usetex'] = True

df = pd.read_csv('bench.tsv', sep='\t')

fig, ax = plt.subplots()
sns.lineplot(x='k', y='norm_acc', data=df, markers=True, label=r'test-acc')
sns.lineplot(x='k', y='adv_acc', data=df, markers=True, label=r'adv-acc')

xs = df['k']
ys = (df['norm_acc'] + df['adv_acc']) / 2
plt.plot(xs,ys,label='total')

ax.set_ylabel('Accuracy')
ax.set_xlabel(r'$k$')
fig.savefig("output.png")