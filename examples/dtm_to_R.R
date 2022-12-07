#
# Example to show how to import a document-term matrix from tmtoolkit/Python into R.
# Python code is available in dtm_to_R.py.
#
# Markus Konrad <post@mkonrad.net>
# Dec. 2022
#

# load libraries
library(Matrix)       # for sparseMatrix in RDS file
library(topicmodels)  # for LDA()
library(slam)         # for as.simple_triplet_matrix()

# load data 
dtm <- readRDS('data/dtm.RDS')
class(dtm)
dtm  # sparse matrix with document labels as row names, vocabulary as column names

# convert sparse matrix to triplet format required for LDA
dtm <- as.simple_triplet_matrix(dtm)

# fit a topic model
topicmodel <- LDA(dtm, k = 20, method = 'Gibbs')

# investigate the topics
terms(topicmodel, 5)
