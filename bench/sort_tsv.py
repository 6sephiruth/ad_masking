import pandas as pd
import seaborn as sns

filename = 'bench_best_by.tsv'
sort_cols = ['q_mask', 'mode', 'attr_method']

df = pd.read_csv(filename, sep='\t')
df.drop_duplicates(inplace=True)
df.sort_values(by=sort_cols, inplace=True)

#df.to_csv('aa.tsv', sep='\t', index=False, float_format='%.4f')