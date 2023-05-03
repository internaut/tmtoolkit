"""
Example to show how to export a document-term matrix from R in order to load it into Python/tmtoolkit.
Python code is available in dtm_to_R.py.

.. codeauthor:: Markus Konrad <post@mkonrad.net>
.. date:: Dec. 2022
"""

import os.path

from tmtoolkit.bow.dtm import read_dtm_from_rds
from tmtoolkit.bow.bow_stats import sorted_terms_table

#%%

rds_file = os.path.join('data', 'dtm2.RDS')
print(f'loading DTM, document labels and vocabulary from file "{rds_file}"')
dtm, doc_labels, vocab = read_dtm_from_rds(rds_file)

#%%

print('first 10 document labels:')
print(doc_labels[:10])

print('first 10 vocabulary tokens:')
print(vocab[:10])

print('DTM shape:')
print(dtm.shape)

#%%

print('top 3 terms per document:')
print(sorted_terms_table(dtm, vocab=vocab, doc_labels=doc_labels, top_n=3))
