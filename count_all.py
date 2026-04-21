import pandas as pd
import os
import glob

def get_count(p):
    for e in ['gb18030', 'utf-8-sig', 'gbk']:
        try:
            df = pd.read_csv(p, encoding=e)
            return len(df), df.iloc[:,3].nunique()
        except:
            continue
    return 0, 0

print("File, Rows, UniqPatients")
# Find all CSV files on G: that start with CT
files = glob.glob('G:/**/*CT*.csv', recursive=True)
for f in files:
    r, p = get_count(f)
    print(f"{os.path.basename(f)}, {r}, {p}")
