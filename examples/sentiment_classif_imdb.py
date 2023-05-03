"""
Example script that loads and processes a large dataset of IMDB movie reviews and then uses naive Bayes and logistic
regression for sentiment classification.

The data comes from [Maas2011]_, available at https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz. It is
not included in this repository, hence you need to download it from there.

This examples requires that you have installed tmtoolkit with the recommended set of packages and have installed an
English language model for spaCy:

    pip install -U "tmtoolkit[recommended]"
    python -m tmtoolkit setup en

For more information, see the installation instructions: https://tmtoolkit.readthedocs.io/en/latest/install.html

.. codeauthor:: Markus Konrad <post@mkonrad.net>
.. date:: Sep. 2022

.. [Maas2011]  Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011).
               Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for
               Computational Linguistics (ACL 2011)
"""

import tarfile
from collections import defaultdict

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from tmtoolkit import corpus as c
from tmtoolkit.bow import NaiveBayesClassifier
from tmtoolkit.utils import enable_logging, indices_of_matches

#%% constants

# download from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz if necessary

DATASET_FILE = 'data/aclImdb_v1.tar.gz'

#%% loading the data

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
                    dataset[ds_type][label][f'{ds_type}_{label}_{file[:-4]}'] = f.read().decode('utf-8')

print(f'training set: loaded {len(dataset["train"]["pos"])} positive and {len(dataset["train"]["neg"])} negative '
      f'samples')
print(f'test set: loaded {len(dataset["test"]["pos"])} positive and {len(dataset["test"]["neg"])} negative samples')

#%% creating a corpus for training and a corpus for testing

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

#%% print initial summaries

for ds_type in ('train', 'test'):
    print(f'initial {ds_type} corpus:')
    c.print_summary(corp[ds_type])

#%% apply token normalization (same for both corpora)

for ds_type in ('train', 'test'):
    print(f'applying token normalization to {ds_type} corpus')
    c.lemmatize(corp[ds_type])
    c.to_lowercase(corp[ds_type])
    c.filter_clean_tokens(corp[ds_type], remove_shorter_than=2, remove_numbers=True)
    c.remove_uncommon_tokens(corp[ds_type], df_threshold=0.01)

#%% print summary again

for ds_type in ('train', 'test'):
    print(f'{ds_type} corpus after token normalization:')
    c.print_summary(corp[ds_type])

#%% train the NB classifier

print('training naive bayes classifier')

nb = NaiveBayesClassifier()

docnames_train = c.doc_labels(corp['train'], sort=False)
pos_docnames_train = [d for d in docnames_train if '_pos_' in d]
neg_docnames_train = [d for d in docnames_train if '_neg_' in d]
assert len(pos_docnames_train) == len(neg_docnames_train) == 12500

nb.fit(corp['train'], classes_docs={
    1: pos_docnames_train,   # class 1 means positive sentiment
    0: neg_docnames_train,   # class 0 means negative sentiment
})

#%% evaluate the NB classifier

print('evaluating naive bayes classifier with test set')

docnames_test = c.doc_labels(corp['test'], sort=False)
docnames_full = docnames_train + docnames_test
pos_docnames_full = {d for d in docnames_full if '_pos_' in d}
assert len(pos_docnames_full) == 25000

y_true = []
y_pred = []
for lbl, d in corp['test'].items():
    y_true.append(int(lbl in pos_docnames_full))
    y_pred.append(nb.predict(d))

#%% show the confusion matrix and classification metrics for NB

print('confusion matrix for pos/neg classes:')
print(metrics.confusion_matrix(y_true, y_pred, labels=[1, 0]))

print('accuracy:', metrics.accuracy_score(y_true, y_pred))
print('precision:', metrics.precision_score(y_true, y_pred, pos_label=1))
print('recall:', metrics.recall_score(y_true, y_pred, pos_label=0))

#%% generate the training data for logistic regression

print('generating training DTM for logistic regression')

dtm_train, doclbls_train, vocab_train = c.dtm(corp['train'], tokens_as_hashes=True, return_doc_labels=True,
                                              return_vocab=True)
scale_factor = 1.0 / np.max(dtm_train)
dtm_train = scale_factor * dtm_train.astype(np.float32)

#%% generate the training labels for logistic regression

print('generating training labels for logistic regression')

y_train = np.array([int(lbl in pos_docnames_full) for lbl in doclbls_train])

#%% fit the log. regr. model

print('fitting logistic regression model')

logreg = LogisticRegression()
logreg.fit(dtm_train, y_train)


#%% generate the evaluation data for log. regr.

print('generating evaluation DTM for logistic regression')

dtm_test, doclbls_test, vocab_test = c.dtm(corp['test'], tokens_as_hashes=True, return_doc_labels=True,
                                           return_vocab=True)
dtm_test = scale_factor * dtm_test.astype(np.float32)   # using same scaling as in training

#%% generate the evaluation labels for log. regr.

print('generating evaluation labels')

y_true = np.array([int(lbl in pos_docnames_full) for lbl in doclbls_test])

#%% prepare the evaluation data

print('preparing evaluation data')

# we need to match the test vocabulary indices with the train vocabulary indices

vocab_test = np.array(vocab_test, dtype='uint64')
vocab_train = np.array(vocab_train, dtype='uint64')

# important note: this assumes that the test vocabulary is smaller than the train vocabulary
# TODO: adapt code to get rid of this assumption
vocab_test_in_train = np.in1d(vocab_test, vocab_train)
vocab_test_subset = vocab_test[vocab_test_in_train]
tok_matching_ind = indices_of_matches(vocab_test_subset, vocab_train)

# important note: this doesn't use sparse matrices, hence it will be heavy on memory
# TODO: adapt code to use sparse matrices
dtm_test_matched_rows = []
for i in range(dtm_test.shape[0]):
    tok_matched = np.zeros(len(vocab_train), dtype='float32')
    tok_matched[tok_matching_ind] = dtm_test[i, vocab_test_in_train].toarray()[0]
    dtm_test_matched_rows.append(tok_matched)

dtm_test_matched = np.vstack(dtm_test_matched_rows)

#%% show the confusion matrix and classification metrics for log. regr.

print('evaluating logistic regression classifier with test set')

y_pred = logreg.predict(dtm_test_matched)

print('confusion matrix for pos/neg classes:')
print(metrics.confusion_matrix(y_true, y_pred, labels=[1, 0]))

print('accuracy:', metrics.accuracy_score(y_true, y_pred))
print('precision:', metrics.precision_score(y_true, y_pred, pos_label=1))
print('recall:', metrics.recall_score(y_true, y_pred, pos_label=1))

print('done.')
