# %%
import pandas as pd

df = pd.read_csv('./data/train.csv')
df.to_csv('temp.csv', index=False)

df_new = pd.read_csv('temp.csv')
print('df_new')
df_new
# %%
print(df_new.columns)
print(df_new.columns.get_loc('Subject Focus'))
print(df_new.columns.get_loc('Blur'))
# print(df_new.columns.get_loc('jaskdfjk')) # Checkingf for nonexistent colnames
# %%
'Pawpularity' in df_new.columns