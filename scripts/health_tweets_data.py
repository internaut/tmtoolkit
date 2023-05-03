"""
Convert health tweets data from https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter to CSV file.

"""

import os
from glob import glob

import pandas as pd

#%%

try:
    ROOT_PATH = os.path.dirname(__file__)
except NameError:
    ROOT_PATH = os.path.join(os.getcwd(), 'scripts')

DATA_DIR = os.path.join(ROOT_PATH, 'fulldata','Health-Tweets')

#%% parsing input data

print(f'loading tweets from {DATA_DIR}')

# all this would not be necessary, had the authors of the dataset used a valid CSV format and the same file encoding
# for all files

USES_WIN_ENCODING = {         # found out with "chardet" utility
    'foxnewshealth.txt',
    'KaiserHealthNews.txt',
    'msnhealthnews.txt',
    'NBChealth.txt',
    'wsjhealth.txt'
}

FIELDS = ['source', 'id', 'date', 'text']
rows = []
for txtfile in glob(os.path.join(DATA_DIR, '*.txt')):
    filebasename = os.path.basename(txtfile)
    print(f'> {filebasename}')
    fileenc = 'Windows-1252' if filebasename in USES_WIN_ENCODING else 'UTF-8'

    with open(txtfile, 'r', encoding=fileenc) as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue

            cur_field = 1
            rowdata = {'source': filebasename[:-4]}
            start = 0
            while cur_field < 4:
                if cur_field < 3:
                    end = line.index('|', start)
                else:
                    end = None
                field_data = line[start:end]
                rowdata[FIELDS[cur_field]] = field_data
                cur_field += 1
                if end is not None:
                    start = end + 1
            assert len(rowdata) == len(FIELDS)
            rows.append(rowdata)

print(f'loaded {len(rows)} tweets')

#%%

df = pd.DataFrame(rows)

output_file = os.path.join(ROOT_PATH, 'tmp', 'health_tweets.csv')
print(f'writing output to {output_file}')
df.to_csv(output_file, index=False)

df['source_id'] = df.source.str.cat(df.id, sep='-')

output_file = os.path.join(ROOT_PATH, 'tmp', 'health_tweets_no_date.csv')
print(f'writing output to {output_file}')
df.loc[:, ['source_id', 'text']].to_csv(output_file, index=False)

print('done.')
