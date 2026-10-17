'''
Summary of Information Tables.
'''

import pandas as pd

base_dir = './'

# The three files below are analysis results generated during an actual run:
#   features.csv     -> plaque features
#   swc_all.csv      -> plaque coordinates / registration
#   cls_correct.csv  -> morphotype classification results
# Adjust the file names to match your own generated files.
f_dir = base_dir + "features.csv"
swc_dir = base_dir + "swc_all.csv"
cls_dir = base_dir + "cls_correct.csv"

df_f = pd.read_csv(f_dir).iloc[:, 1:]
df_xyz = pd.read_csv(swc_dir).iloc[:, 2:]
df_cls = pd.read_csv(cls_dir).iloc[:, 1:]

# sorted_df = pd.merge(df_f, df_xyz, on='ID', sort=False)
sorted_df = pd.merge(pd.merge(df_f, df_xyz, on='ID', sort=False), df_cls, on='ID', sort=False)
# sorted_df.to_csv(base_dir + "unite_all.csv", index=False)

for group in ["multi", "texture", "shape", "gray"]:
    cls_dir = base_dir + group + "/unite_5xcls.csv"
    df_5xcls = pd.read_csv(cls_dir).iloc[:,1:]   # Clustered classification information.
    df_5xcls.columns = [group, 'ID']
    if group == "multi":
        sorted_df2 = pd.merge(sorted_df, df_5xcls, on='ID', sort=False)
    else:
        sorted_df2 = pd.merge(sorted_df2, df_5xcls, on='ID', sort=False)

sorted_df2.to_csv(base_dir + "unite.csv", index=False)



