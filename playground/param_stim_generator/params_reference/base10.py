import pandas as pd


target_csv = 'params_allen_full.csv'
import pdb; pdb.set_trace()
df = pd.read_csv(target_csv)
df['Lower bound'] = df['Base value']/2
df['Upper bound'] = df['Base value']*2
df.to_csv(target_csv, index=False)
# need to remove row at bottom?