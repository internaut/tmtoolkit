library(Matrix)       # for sparseMatrix
library(tm)


data("crude")

dtm <- DocumentTermMatrix(crude, control = list(removePunctuation = TRUE, stopwords = TRUE))

dtm_out <- sparseMatrix(i = dtm$i, j = dtm$j, x = dtm$v, dims = dim(dtm),  dimnames = dimnames(dtm))

saveRDS(dtm_out, 'data/dtm2.RDS')
