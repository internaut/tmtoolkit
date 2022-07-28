import numpy as np
from scipy import sparse
from tmtoolkit.corpus import dtm

#from tmtoolkit.bow.bow_stats import tf_log, tf_binary


class NaiveBayesClassifier:
    def __init__(self, k=1.0):
        self.k = k

        self.token_counts_ = None
        self.classes_ = None
        self.vocab_ = None
        self.prior_ = None

    def fit(self, corp, classes_docs, tokens_as_hashes=True):
        if not classes_docs:
            raise ValueError('at least one class must be given in `classes_docs`')

        dtm_mat, doclbls, vocab = dtm(corp, tokens_as_hashes=tokens_as_hashes, return_doc_labels=True,
                                      return_vocab=True)
        self.vocab_ = np.array(vocab, dtype='uint64' if tokens_as_hashes else 'str')

        doclbls_arr = np.array(doclbls)
        classes_dtm_rows = []
        for c, c_docs in classes_docs.items():
            if not isinstance(c_docs, np.ndarray):
                c_docs = np.array(c_docs)
            c_ind = np.nonzero(c_docs[:, np.newaxis] == doclbls_arr)[1]
            if len(c_ind) > 0:
                # np.sum with axis=0 would be possible but produces dense row; to stick with sparse matrix, compute
                # it as follows:
                c_dtm_row = dtm_mat[c_ind[0], :]
                for i in c_ind[1:]:
                    c_dtm_row += dtm_mat[i, :]

                classes_dtm_rows.append(c_dtm_row)

        class_sizes = np.fromiter(map(len, classes_docs.values()), dtype='float64', count=len(classes_docs))
        self.prior_ = np.log(class_sizes) - np.log(dtm_mat.shape[0])
        self.token_counts_ = sparse.vstack(classes_dtm_rows)
        self.classes_ = list(classes_docs.keys())

        assert len(self.classes_) == self.token_counts_.shape[0]
        assert len(self.vocab_) == self.token_counts_.shape[1]

    def predict(self):
        pass

    def prob(self, tok, classes=None):
        classes = classes or self.classes_
        # TODO
