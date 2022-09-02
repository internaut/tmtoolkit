import tarfile
from collections import defaultdict

from sklearn import metrics
from tmtoolkit import corpus as c
from tmtoolkit.bow import NaiveBayesClassifier
from tmtoolkit.utils import enable_logging

#%%

# https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

DATASET_FILE = 'data/aclImdb_v1.tar.gz'

#%%

print(f'loading dataset from {DATASET_FILE}')

dataset = {'train': defaultdict(dict), 'test': defaultdict(dict)}

with tarfile.open(DATASET_FILE, 'r:gz') as tar:
    members = tar.getmembers()
    n_members = len(members)
    for i, member in enumerate(members, 1):
        if i == 1 or i % 1000 == 0:
            print(f'> member {i} / {n_members}')

        pathcomp = member.path.split('/')
        if len(pathcomp) == 4:
            _, ds_type, label, file = pathcomp
            if ds_type in dataset.keys() and label in {'pos', 'neg'} and file.endswith('.txt'):
                f = tar.extractfile(member)
                if f is not None:
                    dataset[ds_type][label][f'{label}_{file[:-4]}'] = f.read().decode('utf-8')

print(f'training set: loaded {len(dataset["train"]["pos"])} positive and {len(dataset["train"]["neg"])} negative '
      f'samples')
print(f'test set: loaded {len(dataset["test"]["pos"])} positive and {len(dataset["test"]["neg"])} negative samples')

#%%

enable_logging()

corp = {}

for ds_type in ('train', 'test'):
    print(f'generating {ds_type} corpus')
    print(f'> adding positive documents')
    corp[ds_type] = c.Corpus(dataset[ds_type]['pos'], language='en', max_workers=1.0, raw_preproc=c.strip_tags)
    del dataset[ds_type]['pos']
    print(f'> adding negative documents')
    corp[ds_type].update(dataset[ds_type]['neg'])
    del dataset[ds_type]['neg']

del dataset

#%%

for ds_type in ('train', 'test'):
    print(f'initial {ds_type} corpus:')
    c.print_summary(corp[ds_type])

#%%

for ds_type in ('train', 'test'):
    print(f'applying token normalization to {ds_type} corpus')
    c.lemmatize(corp[ds_type])
    c.to_lowercase(corp[ds_type])
    c.filter_clean_tokens(corp[ds_type], remove_shorter_than=2, remove_numbers=True)
    c.remove_uncommon_tokens(corp[ds_type], df_threshold=0.01)

#%%

for ds_type in ('train', 'test'):
    print(f'{ds_type} corpus after token normalization:')
    c.print_summary(corp[ds_type])

#%%

print('training naive bayes classifier')

nb = NaiveBayesClassifier()

docnames = c.doc_labels(corp['train'], sort=False)
pos_docnames = [d for d in docnames if d.startswith('pos_')]
neg_docnames = [d for d in docnames if d.startswith('neg_')]
assert len(pos_docnames) == len(neg_docnames) == 12500

nb.fit(corp['train'], classes_docs={
    'pos': pos_docnames,
    'neg': neg_docnames,
})

#%%

print('evaluating naive bayes classifier with test set')

y_true = []
y_pred = []
for lbl, d in corp['test'].items():
    y_true.append(lbl[:3])
    y_pred.append(nb.predict(d))

#%%

print('confusion matrix for pos/neg classes:')
print(metrics.confusion_matrix(y_true, y_pred, labels=['pos', 'neg']))

print('accuracy:', metrics.accuracy_score(y_true, y_pred))
print('precision:', metrics.precision_score(y_true, y_pred, pos_label='pos'))
print('recall:', metrics.recall_score(y_true, y_pred, pos_label='pos'))

#%%

