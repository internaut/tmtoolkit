"""
N-gram models.

TODO:

  - make sure runs as unigram model, too (N = 1)
  - allow list of sentences as input to `fit`, too
  - add simple translation function between string and hash sequences
  - add __str__ and __repr__ methods

"""

import math
import random
from collections import Counter

from tmtoolkit.corpus import doc_tokens
from tmtoolkit.tokenseq import token_ngrams


OOV = 0
SENT_START = 10
SENT_END = 11


class NGramModel:
    def __init__(self, n, add_k_smoothing=1.0):
        self.n = n
        self.k = add_k_smoothing
        self.vocab_size_ = 0
        self.n_unigrams_ = 0
        self.ngram_counts_ = Counter()
        self.unigram_counts_ = {}
        #self.ngram_prob_ = {}  # np.array([], dtype=np.float_)

    def fit(self, corp, tokens_as_hashes=True):
        unigram_sents = []
        for sents in doc_tokens(corp, tokens_as_hashes=tokens_as_hashes, sentences=True).values():
            new_sents = []
            for s in sents:
                new_sents.append(self.pad_sequence(s))
            unigram_sents.extend(new_sents)

        self.ngram_counts_ = Counter()
        for i in range(1, self.n+1):
            ngrms_i = []
            for sent in unigram_sents:
                if i == 1:
                    ngrms_i.extend(map(lambda x: (x, ), sent))
                else:
                    ngrms_i.extend(token_ngrams(sent, n=i, join=False, ngram_container=tuple))
            self.ngram_counts_.update(ngrms_i)

        self.unigram_counts_ = [c for ng, c in self.ngram_counts_.items() if len(ng) == 1]
        self.vocab_size_ = len(self.unigram_counts_)
        self.n_unigrams_ = sum(self.unigram_counts_)

    def predict(self, given=None, return_prob=0):
        """
        Predict the most likely next token given a sequence of tokens `given`. If `given` is None, assume a sentence
        start.

        :param given: given sequence of tokens; if None, assume a sentence start
        :param return_prob: 0 - don't return prob., 1 – return prob., 2 – return log prob.
        :return: if `return_prob` is 0, return the most likely next token; if `return_prob` is not zero, return a
                 2-tuple with ``(must likely token, predition probability)``
        """
        given = self._prepare_given_param(given)
        probs = self._probs_for_given(given, log=return_prob==2)

        if probs:
            probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            if return_prob != 0:
                return probs[0]
            else:
                return probs[0][0]
        else:
            return None

    def generate_sequence(self, given=None, until_n=None, until_token=SENT_END):
        given = self._prepare_given_param(given)

        i = 0
        while True:
            probs = self._probs_for_given(given, log=False)

            if not probs:
                break

            x = random.choices(list(probs.keys()), list(probs.values()))[0]
            given = given[1:] + (x, )
            i += 1

            yield x

            if until_n is not None and i >= until_n:
                break

            if until_token is not None and x == until_token:
                break

    def prob(self, x, given=None, log=True, pad_input=False):
        if isinstance(x, list):
            x = tuple(x)

        if isinstance(given, list):
            given = tuple(given)

        if not isinstance(x, tuple):
            x = (x,)

        if given is not None:
            if not isinstance(given, tuple):
                given = (given,)
            x = given + x

        if pad_input:
            x = self.pad_sequence(x)

        if len(x) > self.n:
            x = token_ngrams(x, self.n, join=False, ngram_container=tuple)
        else:
            x = [x]

        p = 0 if log else 1
        for ng in x:
            p_ng = self._prob_smooth(ng, log=log)
            if log:
                p += p_ng
            else:
                p *= p_ng

        if log:
            assert 0 <= math.exp(p) <= 1, 'smoothed prob. must be in [0, 1] interval'
        else:
            assert 0 <= p <= 1, 'smoothed prob. must be in [0, 1] interval'

        return p

    def perplexity(self, x, pad_input=False):
        if self.vocab_size_ <= 0:
            raise ValueError('vocabulary must be non-empty')

        log_p = self.prob(x, pad_input=pad_input)
        return math.pow(math.exp(log_p), -1.0/self.vocab_size_)

    def pad_sequence(self, s):
        if not isinstance(s, (tuple, list)):
            raise ValueError('`s` must be tuple or list')

        pad = self.n - 1

        if s:
            if isinstance(s, tuple):
                s = list(s)

            s_ = [SENT_START] * pad + s + [SENT_END] * pad

            if isinstance(s, tuple):
                return tuple(s_)
            else:
                return s_
        else:
            if isinstance(s, tuple):
                return tuple()
            else:
                return []

    def _prepare_given_param(self, given):
        if given is None:
            given = (SENT_START, ) * (self.n - 1)
        else:
            if isinstance(given, list):
                given = tuple(given)
            elif not isinstance(given, tuple):
                given = (given,)

            if len(given) > self.n - 1:
                given = given[-(self.n - 1):]
            elif len(given) < self.n - 1:
                raise ValueError(f'for a {self.n}-gram model you must provide `given` with at least {self.n-1} tokens')

        assert len(given) == self.n - 1

        return given

    def _prob_smooth(self, x, log):
        n = len(x)
        assert isinstance(x, tuple), '`x` must be a tuple'
        assert 1 <= n <= self.n, f'`x` must be a tuple of length 1 to {self.n} in a {self.n}-gram model'

        c = self.ngram_counts_.get(x, 0)

        if n == 1:  # single token
            d = self.n_unigrams_
        else:       # x[:(self.n-1)] is the "given" sequence, i.e. the sequence before x[-1]
            d = self.ngram_counts_.get(x[:(self.n-1)], 0)

        if log:
            p = math.log(c + self.k) - math.log(d + self.k * self.vocab_size_)
            assert 0 <= math.exp(p) <= 1, 'smoothed prob. must be in [0, 1] interval'
        else:
            p = (c + self.k) / (d + self.k * self.vocab_size_)
            assert 0 <= p <= 1, 'smoothed prob. must be in [0, 1] interval'

        return p

    def _probs_for_given(self, given, log):
        probs = {}
        len_g = len(given)
        for ng in self.ngram_counts_.keys():
            if len(ng) == len_g + 1 and ng[:len_g] == given:
                candidate = ng[len_g:]
                assert len(candidate) == 1
                assert candidate not in probs
                probs[candidate[0]] = self._prob_smooth(ng, log=log)

        return probs

