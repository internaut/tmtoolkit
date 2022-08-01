import numpy as np
from scipy import sparse
from tmtoolkit.corpus import dtm

#from tmtoolkit.bow.bow_stats import tf_log, tf_binary
from tmtoolkit.utils import indices_of_matches


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

    def predict(self, tok, return_prob=0):
        probs = self.prob(tok)
        i_max = np.argmax(probs)
        c_max = self.classes_[i_max]

        if return_prob > 0:
            p = probs[i_max]
            if return_prob == 2:
                p = np.exp(p)
            return c_max, p
        else:
            return c_max

    def prob(self, tok, classes=None, log=True):
        if classes is None:
            classes = self.classes_

        if not classes:
            return np.array([])

        if not isinstance(tok, (list, tuple)):
            tok = [tok]

        tok = np.array(tok, dtype=self.vocab_.dtype)
        tok = tok[np.in1d(tok, self.vocab_)]

        probs = []
        for c in classes:
            if c not in self.classes_:
                raise ValueError(f'unknown class: {c}')
            i_c = self.classes_.index(c)
            tok_ind = indices_of_matches(tok, self.vocab_)
            c_counts = self.token_counts_[i_c, :]
            tok_c = c_counts[0, tok_ind]

            p = self.prior_[i_c] \
                + np.sum(np.log(tok_c.todense() + self.k) - np.log(np.sum(c_counts.todense() + self.k)))

            if log:
                probs.append(p)
            else:
                probs.append(np.exp(p))

        return np.array(probs)
