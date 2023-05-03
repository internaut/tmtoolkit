"""
An example to show how to export prepare and export a document-term matrix to R.
R code is available in dtm_to_R.R.

.. codeauthor:: Markus Konrad <post@mkonrad.net>
.. date:: Dec. 2022
"""
import os.path

import tmtoolkit.corpus as c
from tmtoolkit.bow.dtm import save_dtm_to_rds

#%%

# load built-in sample dataset and use all possible worker processes
corp = c.Corpus.from_builtin_corpus('en-News100', max_workers=1.0)
c.print_summary(corp)

#%%

# apply some text normalization
c.lemmatize(corp)
c.to_lowercase(corp)
c.filter_clean_tokens(corp, remove_numbers=True)
c.remove_common_tokens(corp, df_threshold=0.90)
c.remove_uncommon_tokens(corp, df_threshold=0.05)
c.remove_documents_by_length(corp, '<', 30)

c.print_summary(corp)

#%%

# build sparse document-token matrix (DTM)
# document labels identify rows, vocabulary tokens identify columns
mat, doc_labels, vocab = c.dtm(corp, return_doc_labels=True, return_vocab=True)

#%%

print('first 10 document labels:')
print(doc_labels[:10])

print('first 10 vocabulary tokens:')
print(vocab[:10])

print('DTM shape:')
print(mat.shape)

#%%

rds_file = os.path.join('data', 'dtm.RDS')
print(f'saving DTM, document labels and vocabulary to file "{rds_file}"')
save_dtm_to_rds(rds_file, mat, doc_labels, vocab)

print('done.')
