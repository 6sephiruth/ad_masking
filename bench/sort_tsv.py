import pandas as pd
import seaborn as sns

filename = 'bench_mnist.tsv'
sort_cols = ['q_mask', 'mode', 'attr_method']

df = pd.read_csv(filename, sep='\t')
df.drop_duplicates(inplace=True)
df.sort_values(by=sort_cols, inplace=True)

df.to_csv(filename, sep='\t', index=False, float_format='%.4f')