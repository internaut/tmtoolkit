"""
N-gram models.

TODO:

  - make sure runs as unigram model, too (N = 1)

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
    def __init__(self, n):
        self.n = n
        self.ngram_counts_ = Counter()
        self.ngram_prob_ = {}  # np.array([], dtype=np.float_)

    def fit(self, corp):
        unigram_sents = []
        for sents in doc_tokens(corp, tokens_as_hashes=True, sentences=True).values():
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

        self.ngram_prob_ = {}
        n_unigrams = sum(c for ng, c in self.ngram_counts_.items() if len(ng) == 1)
        for ng, c in self.ngram_counts_.items():
            n = len(ng)
            given = ng[:(n - 1)]

            if n == 1:
                p = math.log(c) - math.log(n_unigrams)
                lookup = ng[-1]
            else:
                p = math.log(c) - math.log(self.ngram_counts_[given])
                lookup = (ng[-1], given)

            #assert 0 < p <= 1
            self.ngram_prob_[lookup] = p

    def predict(self, given=None, return_prob=0):
        """
        TODO

        :param given:
        :param return_prob: 0 - don't return prob., 1 – return prob., 2 – return log prob.
        :return:
        """
        given = self._prepare_given_param(given)
        probs = self._probs_for_given(given)

        if probs:
            probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            if return_prob == 1:
                return probs[0][0], math.exp(probs[0][1])
            elif return_prob == 2:
                return probs[0]
            else:
                return probs[0][0]
        else:
            return None

    def generate_sequence(self, given=None, until_n=None, until_token=SENT_END):
        given = self._prepare_given_param(given)

        i = 0
        while True:
            probs = self._probs_for_given(given)

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
        elif not isinstance(given, tuple) and given is not None:
            given = (given,)

        if isinstance(x, tuple):
            if given is not None:
                x = given + x
            if pad_input:
                x = self.pad_sequence(x)

            x = token_ngrams(x, self.n, join=False, ngram_container=tuple)

            p = 0 if log else 1
            for ng in x:
                given = ng[:(self.n - 1)]
                log_p = self.ngram_prob_[(ng[-1], given)]
                if log:
                    p += log_p
                else:
                    p *= math.exp(log_p)
            return p
        else:
            if given is None:
                p = self.ngram_prob_[x]
            else:
                p = self.ngram_prob_[(x, given)]

            if log:
                return p
            else:
                return math.exp(p)

    def perplexity(self, x, pad_input=False):
        log_p = self.prob(x, pad_input=pad_input)
        n = sum(len(ng) == 1 for ng in self.ngram_counts_.keys()) - 1
        return math.pow(math.exp(log_p), -1.0/n)

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

    def _probs_for_given(self, given):
        return {tok_given[0]: math.exp(p) for tok_given, p in self.ngram_prob_.items()
                if isinstance(tok_given, tuple) and tok_given[1] == given}

