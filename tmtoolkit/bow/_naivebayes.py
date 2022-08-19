"""
Naive Bayes classifier model as in [JurafskyMartin2021]_, chapt. 4. Mainly provides the :class:`NaiveBayesClassifier`.
class.

TODO:
  - add update method

.. [JurafskyMartin2021] Jurafsky, D. and Martin, J.H., 2021. Speech and Language Processing (3rd ed. draft). Online:
                        https://web.stanford.edu/~jurafsky/slp3/
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Optional, List, Union, Dict, Tuple, Sequence

import numpy as np
from scipy import sparse

if not any(find_spec(pkg) is None for pkg in ('spacy', 'bidict', 'loky')):   # if textproc dep. are installed
    from tmtoolkit.corpus import Corpus, Document
else:
    class Corpus:
        pass
    class Document:
        pass

from tmtoolkit.bow.bow_stats import tf_binary
from tmtoolkit.types import StrOrInt
from tmtoolkit.utils import indices_of_matches, empty_chararray, chararray_elem_size


class NaiveBayesClassifier:
    """
    A Naive Bayes classifier model with smoothing constant `k` based on a sparse class-term frequency matrix.
    """

    def __init__(self, add_k_smoothing: float = 1.0, binary_counts: bool = False, tokens_as_hashes: bool = True):
        """
        Initialize a Naive Bayes classifier model with smoothing constant `add_k_smoothing`. Optionally use binary
        counts in document-term matrix.

        :param add_k_smoothing: smoothing constant added to each count; must be positive
        :param binary_counts: if True, use binary counts in document-term matrix
        :param tokens_as_hashes: if True, use token type hashes (integers) instead of textual representations (strings)
        """

        if add_k_smoothing < 0:
            raise ValueError('`add_k_smoothing` must be positive')

        self.k: float = add_k_smoothing
        self.binary_counts: bool = binary_counts
        self.tokens_as_hashes: bool = tokens_as_hashes

        # a sparse matrix of shape (C, V), where C is the number of classes and V is the vocabulary size;
        # stores the frequency of tokens per class (class-term frequency matrix)
        self.token_counts_: Optional[sparse.csr_matrix] = None

        # the list of classes this model was trained on
        self.classes_: Optional[List[str]] = None

        # the size of each class in `self.classes_` (i.e. num. of documents in each class)
        self.class_sizes_: Optional[np.ndarray] = None

        # the token vocabulary
        self.vocab_: Optional[np.ndarray] = None

        # the log prior; has shape (C, ) where C is the number of classes; stores log prob. of each class
        self.prior_: Optional[np.ndarray] = None

    def __str__(self) -> str:
        """String representation of this NaiveBayesClassifier object."""
        return self.__repr__()

    def __repr__(self):
        """String representation of this NaiveBayesClassifier object."""
        return f'<NaiveBayesClassifier [k={self.k}, binary_counts={self.binary_counts}]>'

    @property
    def n_trained_docs(self) -> int:
        """
        The number of documents used to train this classifier.

        :return: number of documents used to train this classifier
        """
        return np.sum(self.class_sizes_)

    def fit(self, data: Union[Corpus, Tuple[Union[sparse.csr_matrix, np.ndarray], Sequence, Sequence]],
            classes_docs: Dict[str, Union[list, tuple, np.ndarray]]) -> NaiveBayesClassifier:
        """
        Fit this naive bayes model using a :class:`~tmtoolkit.corpus.Corpus` object and a dict `classes_docs` that
        maps classes to document labels.

        :param data: either a :class:`~tmtoolkit.corpus.Corpus` object with training documents or tuple consisting of a
                     (sparse) document-term matrix, document labels and vocabulary
        :param classes_docs: a dict that maps classes to document labels or indices
        :return: this instance
        """

        dtm_mat, doclbls, vocab = self._prepare_passed_data(data, classes_docs)

        # store vocab as numpy array
        self.vocab_ = np.array(vocab, dtype='uint64' if self.tokens_as_hashes else 'str')

        # get token counts
        self.token_counts_ = self._generate_token_counts(dtm_mat, doclbls, classes_docs)

        # calculate the log prior, i.e. the prob. that a document is part of class `c`: log(N_c / N_docs)
        self.class_sizes_ = self._class_sizes_array(classes_docs)
        self._update_prior()

        # store the classes
        self.classes_ = list(classes_docs.keys())

        assert len(self.classes_) == len(self.class_sizes_) == self.token_counts_.shape[0]
        assert len(self.vocab_) == self.token_counts_.shape[1]

        return self

    def update(self, data: Union[Corpus, Tuple[Union[sparse.csr_matrix, np.ndarray], Sequence, Sequence]],
               classes_docs: Dict[str, Union[list, tuple, np.ndarray]]) -> NaiveBayesClassifier:

        self._check_fitted()

        dtm_mat, doclbls, vocab = self._prepare_passed_data(data, classes_docs)

        # update classes
        new_classes = set(classes_docs.keys()) - set(self.classes_)
        if len(new_classes) > 0:
            self.classes_.extend(new_classes)

        # update vocabulary
        new_vocab = np.array(list(set(vocab) - set(self.vocab_)), dtype='uint64' if self.tokens_as_hashes else 'str')
        if len(new_vocab) > 0:
            if self.tokens_as_hashes:
                self.vocab_ = np.concatenate((self.vocab_, new_vocab))
            else:
                old_vocab_charlen = chararray_elem_size(self.vocab_)
                new_vocab_charlen = max(old_vocab_charlen, chararray_elem_size(new_vocab))
                if new_vocab_charlen > old_vocab_charlen:
                    self.vocab_ = np.concatenate((self.vocab_.astype('<U' + str(new_vocab_charlen)), new_vocab))
                else:
                    self.vocab_ = np.concatenate((self.vocab_, new_vocab))

        # get token counts for update
        new_token_counts = self._generate_token_counts(dtm_mat, doclbls, classes_docs)

        # update the token counts
        doc_indices = indices_of_matches(np.array(list(classes_docs.keys())), np.array(self.classes_))
        tok_indices = indices_of_matches(np.array(vocab), self.vocab_)

        assert new_token_counts.shape == (len(doc_indices), len(tok_indices))

        if len(new_classes) > 0 or len(new_vocab) > 0:   # new classes or token types added -> resize matrix
            m, n = self.token_counts_.shape
            self.token_counts_.resize((m + len(new_classes), n + len(new_vocab)))

        self.token_counts_[doc_indices[:, np.newaxis], tok_indices] += new_token_counts

        # update the log prior
        if len(new_classes) > 0:   # new classes added -> expand class sizes array
            self.class_sizes_ = np.concatenate((self.class_sizes_,
                                                np.zeros(len(new_classes), dtype=self.class_sizes_.dtype)))

        new_class_sizes = self._class_sizes_array(classes_docs)
        self.class_sizes_[doc_indices] += new_class_sizes
        self._update_prior()

        assert len(self.classes_) == len(self.class_sizes_) == self.token_counts_.shape[0]
        assert len(self.vocab_) == self.token_counts_.shape[1]

        return self

    def predict(self, tok: Union[StrOrInt, List[StrOrInt], Tuple[StrOrInt, ...], np.ndarray, Document],
                return_prob: int = 0) -> Union[str, Tuple[str, float]]:
        """
        Predict the most likely class for a :class:~`tmtoolkit.corpus.Document` object or sequence of tokens `tok`.
        Dismisses all tokens in `tok` that are not present in the trained vocabulary.

        :param tok: :class:~`tmtoolkit.corpus.Document` object or sequence of tokens `tok` or single token
        :param return_prob: 0 - don't return prob., 1 – return prob., 2 – return log prob.
        :return: if `return_prob` is 0, return the most likely class; if `return_prob` is not zero, return a
                 2-tuple with ``(must likely class, prediction probability)`` where the prob. is given as log prob.
                 if `return_prob` is 2
        """

        # get prediction probabilities
        probs = self.prob(tok)

        # get index of max. prob.
        i_max = np.argmax(probs)

        # get class for max. prob.
        c_max = self.classes_[i_max]

        if return_prob > 0:
            p = probs[i_max]
            if return_prob == 1:
                p = np.exp(p)
            return c_max, p
        else:
            return c_max

    def prob(self, tok: Union[StrOrInt, List[StrOrInt], Tuple[StrOrInt, ...], np.ndarray, Document],
             classes: Optional[Union[List[str], Tuple[str, ...]]] = None, log: bool = True) -> np.ndarray:
        """
        Return the probability of a :class:~`tmtoolkit.corpus.Document` object or sequence of tokens `tok` for each
        class in `classes` or for all classes of this model if `classes` is None. Dismisses all tokens in `tok` that are
        not present in the trained vocabulary.

        :param tok: :class:~`tmtoolkit.corpus.Document` object or sequence of tokens `tok` or single token
        :param classes: classes for which the prob. should be determined; if `classes` is None, use all classes of this
                        model, i.e. :attr:`classes_`
        :param log: if True, return log probability
        :return: array with probability of given `tok` for each class in `classes` or for all classes of this model;
                 this array is aligned to `classes` or :attr:`classes_`
        """

        self._check_fitted()

        if classes is None:
            classes = self.classes_

        if not classes:
            return np.array([])

        if isinstance(tok, Document):
            from tmtoolkit.corpus import document_token_attr
            tok = document_token_attr(tok, 'token', as_hashes=self.tokens_as_hashes, as_array=True)
        elif not isinstance(tok, (list, tuple)):
            tok = [tok]

        if not isinstance(tok, np.ndarray):
            tok = np.array(tok, dtype=self.vocab_.dtype)

        # dismiss all tokens in `tok` that are out of vocab.
        tok = tok[np.in1d(tok, self.vocab_)]

        # calculate prob. for each class `c` in `classes`
        probs = []
        for c in classes:
            if c not in self.classes_:
                raise ValueError(f'unknown class: {c}')

            # index of the class
            i_c = self.classes_.index(c)

            # vocab. indices of tokens
            tok_ind = indices_of_matches(tok, self.vocab_)

            # counts of all tokens in this class
            c_counts = self.token_counts_[i_c, :]

            # counts of input document tokens in this class
            tok_c = c_counts[0, tok_ind]

            # probability of each token `t_i` given class `c` as
            #   P(t_i|c) = P(c) * (N_{c,i} + k) / sum_{j,c}[(N_{c,j} + k)],
            # where `N_{c,i}` is the frequency of token `i` from `tok` in class `c` and `N_{c,j}` is the frequency of
            # token `j` in class `c` from all tokens in the vocabulary; `k` is the smoothing constant
            # add up the log probabilities for all tokens `t_i` in `tok`
            p = self.prior_[i_c] \
                + np.sum(np.log(tok_c.todense() + self.k) - np.log(np.sum(c_counts.todense() + self.k)))

            if log:
                probs.append(p)
            else:
                probs.append(np.exp(p))

        return np.array(probs)

    def _prepare_passed_data(self, data: Union[Corpus, Tuple[Union[sparse.csr_matrix, np.ndarray], Sequence, Sequence]],
                             classes_docs: Dict[str, Union[list, tuple, np.ndarray]]) \
            -> Tuple[sparse.csr_matrix, List[str], List[str]]:
        if not classes_docs:
            raise ValueError('at least one class must be given in `classes_docs`')

        # generate a sparse document-term matrix from the corpus
        if isinstance(data, Corpus):
            from tmtoolkit.corpus import dtm
            dtm_mat, doclbls, vocab = dtm(data, tokens_as_hashes=self.tokens_as_hashes, return_doc_labels=True,
                                          return_vocab=True)
        else:
            dtm_mat, doclbls, vocab = data

            if not sparse.issparse(dtm_mat):
                dtm_mat = sparse.csr_matrix(dtm_mat)

        if self.binary_counts:  # optionally transform to binary matrix
            dtm_mat = tf_binary(dtm_mat)

        return dtm_mat, doclbls, vocab

    @staticmethod
    def _generate_token_counts(dtm_mat: sparse.csr_matrix, doclbls: List[str],
                               classes_docs: Dict[str, Union[list, tuple, np.ndarray]]) -> sparse.csr_matrix:
        # count the term frequencies per class
        doclbls_arr = np.array(doclbls)
        classes_dtm_rows = []
        for c, c_docs in classes_docs.items():  # iterate through classes along with their documents
            if not isinstance(c_docs, np.ndarray):
                c_docs = np.array(c_docs) if c_docs else empty_chararray()

            # get the indices of the documents
            try:
                c_ind = indices_of_matches(c_docs, doclbls_arr, check_a_in_b=True)
            except ValueError:
                raise ValueError(f'some document labels for class {c} do not exist in the corpus')

            if len(c_ind) > 0:
                # now count the token frequencies for all documents in this class
                # np.sum with axis=0 would be possible but produces dense row; to stick with sparse matrix, compute
                # it as follows:
                c_dtm_row = dtm_mat[c_ind[0], :]
                for i in c_ind[1:]:
                    c_dtm_row += dtm_mat[i, :]

                classes_dtm_rows.append(c_dtm_row)

        # create the class-term frequency matrix by stacking the rows of each class
        return sparse.vstack(classes_dtm_rows)

    @staticmethod
    def _class_sizes_array(classes_docs: Dict[str, Union[list, tuple, np.ndarray]]) -> np.ndarray:
        return np.fromiter(map(len, classes_docs.values()), dtype='int', count=len(classes_docs))

    def _update_prior(self):
        self.prior_ = np.log(self.class_sizes_) - np.log(self.n_trained_docs)

    def _check_fitted(self):
        if self.token_counts_ is None:
            raise ValueError('the model needs to be fitted before calling this method')


