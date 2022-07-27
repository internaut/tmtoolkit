"""
N-gram models.

TODO:

  - add simple translation function between string and hash sequences
  - add docs
  - add tests

"""

import math
import random
from collections import Counter

from bidict import bidict

from tmtoolkit.corpus import doc_tokens, Corpus
from tmtoolkit.tokenseq import token_ngrams
from tmtoolkit.utils import flatten_list

OOV = 0
SENT_START = 10
SENT_END = 11


SPECIAL_TOKENS = bidict({
    SENT_START: '<s>',
    SENT_END: '</s>',
    OOV: '<oov>'
})


class NGramModel:
    def __init__(self, n, add_k_smoothing=1.0, keep_vocab=None, tokens_as_hashes=True):
        if not isinstance(n, int) or n < 1:
            raise ValueError('`n` must be a strictly positive integer')

        if add_k_smoothing < 0:
            raise ValueError('`add_k_smoothing` must be positive')

        if keep_vocab is not None:
            if not isinstance(keep_vocab, (float, int)):
                raise ValueError('`keep_vocab` must be either a float or an int')

            if keep_vocab <= 0:
                raise ValueError('if `keep_vocab` is given, it must be strictly positive')

            if isinstance(keep_vocab, float) and keep_vocab > 1.0:
                raise ValueError('if `keep_vocab` is given as float, it must be in range (0, 1]')

        self.n = n
        self.k = add_k_smoothing
        self.keep_vocab = keep_vocab
        self.tokens_as_hashes = tokens_as_hashes

        self.vocab_size_ = 0
        self.n_unigrams_ = 0
        self.ngram_counts_ = Counter()

    def __str__(self) -> str:
        """String representation of this NGramModel object"""
        return self.__repr__()

    def __repr__(self):
        return f'<NGramModel [n={self.n}, k={self.k}]>'

    def fit(self, corp):
        if isinstance(corp, Corpus):
            corp = flatten_list(doc_tokens(corp, tokens_as_hashes=self.tokens_as_hashes, sentences=True).values())
        elif not isinstance(corp, list):
            raise ValueError('`corp` must be either a Corpus object or a list of sentences as token sequences')

        unigram_sents = list(map(self.pad_sequence, corp))

        unigram_counts = Counter(t for s in unigram_sents for t in s)

        if self.keep_vocab is not None:
            if isinstance(self.keep_vocab, float):
                keep_n = round(len(unigram_counts) * self.keep_vocab)
            else:
                keep_n = self.keep_vocab
            keep_tok = set(list(zip(*unigram_counts.most_common(keep_n)))[0])
            unigram_counts = {k: v for k, v in unigram_counts.items() if k in keep_tok}
        else:
            keep_tok = None

        self.vocab_size_ = len(unigram_counts)
        self.n_unigrams_ = sum(unigram_counts.values())

        self.ngram_counts_ = Counter()
        oov_tok = OOV if self.tokens_as_hashes else SPECIAL_TOKENS[OOV]
        for i in range(1, self.n+1):
            ngrms_i = []
            for sent in unigram_sents:
                if keep_tok:
                    sent = [t if t in keep_tok else oov_tok for t in sent]
                ngrms_i.extend(token_ngrams(sent, n=i, join=False, ngram_container=tuple))
            self.ngram_counts_.update(ngrms_i)

    def predict(self, given=None, return_prob=0):
        """
        Predict the most likely next token given a sequence of tokens `given`. If `given` is None, assume a sentence
        start.

        :param given: given sequence of tokens; if None, assume a sentence start
        :param return_prob: 0 - don't return prob., 1 – return prob., 2 – return log prob.
        :return: if `return_prob` is 0, return the most likely next token; if `return_prob` is not zero, return a
                 2-tuple with ``(must likely token, predition probability)``
        """
        if not self.ngram_counts_:
            raise ValueError('the model needs to be fitted before calling this method')

        given = self._prepare_given_param(given)
        probs = self._probs_for_given(given, log=return_prob == 2)

        if probs:
            probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            if return_prob != 0:
                return probs[0]
            else:
                return probs[0][0]
        else:
            return None

    def generate_sequence(self, given=None, until_n=None, until_token=SENT_END):
        if not self.ngram_counts_:
            raise ValueError('the model needs to be fitted before calling this method')

        if not self.tokens_as_hashes and isinstance(until_token, int):
            until_token = SPECIAL_TOKENS[until_token]

        given = self._prepare_given_param(given)

        i = 0
        while True:
            probs = self._probs_for_given(given, log=False, backoff=True)

            if not probs:
                break

            x = random.choices(list(probs.keys()), list(probs.values()))[0]
            given = (given + (x, ))[1:]
            i += 1

            yield x

            if until_n is not None and i >= until_n:
                break

            if until_token is not None and x == until_token:
                break

    def prob(self, x, given=None, log=True, pad_input=False):
        if not self.ngram_counts_:
            raise ValueError('the model needs to be fitted before calling this method')

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
        if not self.ngram_counts_:
            raise ValueError('the model needs to be fitted before calling this method')

        if self.vocab_size_ <= 0:
            raise ValueError('vocabulary must be non-empty')

        log_p = self.prob(x, pad_input=pad_input)
        return math.pow(math.exp(log_p), -1.0/self.vocab_size_)

    def pad_sequence(self, s):
        if not isinstance(s, (tuple, list)):
            raise ValueError('`s` must be tuple or list')

        pad = max(self.n - 1, 1)
        start_symbol = SENT_START if self.tokens_as_hashes else SPECIAL_TOKENS[SENT_START]
        end_symbol = SENT_END if self.tokens_as_hashes else SPECIAL_TOKENS[SENT_END]

        if s:
            if isinstance(s, tuple):
                s = list(s)

            s_ = [start_symbol] * pad + s + [end_symbol] * pad

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
        if self.n == 1:
            return tuple()

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

    def _probs_for_given(self, given, log, backoff=False):
        probs = {}
        len_g = len(given)

        while len_g >= 0:
            for ng in self.ngram_counts_.keys():
                if len(ng) == len_g + 1 and ng[:len_g] == given:
                    candidate = ng[len_g:]
                    assert len(candidate) == 1
                    assert candidate not in probs
                    probs[candidate[0]] = self._prob_smooth(ng, log=log)

            if probs or not backoff:
                break

            given = given[1:]
            len_g = len(given)

        return probs

