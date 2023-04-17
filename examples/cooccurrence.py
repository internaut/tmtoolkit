"""
Example script showing the usage of token cooccurrence matrices.

.. codeauthor:: Markus Konrad <post@mkonrad.net>
.. date:: Feb. 2023
"""

import pandas as pd
from tmtoolkit import corpus as c
from tmtoolkit.tokenseq import ppmi
from tmtoolkit.utils import pairwise_max_table

pd.set_option('display.width', 140)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

#%%

print('loading corpus')

corp = c.Corpus.from_builtin_corpus('en-parlspeech-v2-sample-houseofcommons',
                                    max_workers=1.0)

print()
c.print_summary(corp)
print()

print('preprocessing corpus')

c.lemmatize(corp)
c.to_lowercase(corp)

c.filter_clean_tokens(corp,
                      remove_punct=True,
                      remove_stopwords=True,
                      remove_empty=True,
                      remove_numbers=True)
print()
c.print_summary(corp)
print()

#%%

print('generating cooccurrence matrix for whole corpus')

cooc_mat, cooc_tokens = c.token_cooccurrence(corp, context_size=5, return_tokens=True)
pairs_table = pairwise_max_table(cooc_mat, labels=cooc_tokens)

print('top pairs for whole corpus:')
print(pairs_table)

#%%

print('applying PPMI')

print('top pairs for whole corpus by PPMI:')
pairwise_max_table(ppmi(cooc_mat), labels=cooc_tokens)

#%%

n_sample = 10

print(f'generating cooccurrence matrix for the first {n_sample} documents in the corpus')

corp.max_workers = 1
cooc_per_doc, cooc_tokens = c.token_cooccurrence(corp, select=c.doc_labels(corp)[:n_sample],
                                                 context_size=5, per_document=True, triu=True,
                                                 return_tokens=True)


#%%

print('top pairs for each document:')

for lbl, cooc in cooc_per_doc.items():
    print(f'{lbl}:')
    print(c.doc_texts(corp, select=lbl, collapse=' ', n_tokens=20)[lbl], '...\n')
    print(pairwise_max_table(cooc, labels=cooc_tokens, skip_zeros=True).head(5))
    print('---\n')

#%%

print('done.')