#
# Example to show how to export a document-term matrix from R in order to load it into Python/tmtoolkit.
# Python code is available in dtm_to_R.py.
#
# Markus Konrad <post@mkonrad.net>
# Dec. 2022
#

library(Matrix)       # for sparseMatrix
library(tm)           # for DocumentTermMatrix


data("crude")

dtm <- DocumentTermMatrix(crude, control = list(removePunctuation = TRUE, stopwords = TRUE))

dtm_out <- sparseMatrix(i = dtm$i, j = dtm$j, x = dtm$v, dims = dim(dtm),  dimnames = dimnames(dtm))

saveRDS(dtm_out, 'data/dtm2.RDS')
