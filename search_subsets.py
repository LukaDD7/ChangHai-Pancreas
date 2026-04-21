import pandas as pd
import os
import glob

def search_strings(p, targets):
    for e in ['gb18030', 'utf-8-sig', 'gbk']:
        try:
            df = pd.read_csv(p, encoding=e)
            found = {}
            for t in targets:
                mask = df.apply(lambda x: x.astype(str).str.contains(t, case=False, na=False)).any(axis=1)
                found[t] = int(mask.sum())
            return found, df.columns.tolist()
        except:
            continue
    return {}, []

targets = ['SJJR', 'XLA', '神经内分泌', '腺瘤']
files = glob.glob('G:/**/*CT*.csv', recursive=True)

for f in files:
    found, cols = search_strings(f, targets)
    if any(found.values()):
        print(f"File: {os.path.basename(f)}")
        print(f"  Found: {found}")
        source_cols = [c for c in cols if '源' in c]
        if source_cols:
            print(f"  Source Columns: {source_cols}")
