import math
from collections import Counter

from tmtoolkit.corpus import Corpus, doc_tokens
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




# corp = Corpus.from_builtin_corpus('en-parlspeech-v2-sample-houseofcommons', sample=10, max_workers=1.0)
# corp

corp = Corpus({'d1': 'I am Sam', 'd2': 'Sam I am', 'd3': 'I do not like green eggs and ham'}, language='en')
corp

ngmodel = NGramModel(2)
ngmodel.fit(corp)

h = lambda x: corp.nlp.vocab.strings[x]
hsent = lambda x: tuple(h(t) for t in x.split())

#ngmodel.prob(h('I'), SENT_START, log=False)
#ngmodel.prob(h('Sam'), h('am'), log=False)
#ngmodel.prob(h('Sam'), hsent('I am'), log=False)
ngmodel.prob(hsent('I am Sam'), log=False)


ngmodel.ngram_prob_[corp.nlp.vocab.strings['I'], (SENT_START, )]
ngmodel.ngram_prob_[SENT_END, (corp.nlp.vocab.strings['Sam'], )]
ngmodel.ngram_prob_[corp.nlp.vocab.strings['Sam'], (SENT_START, )]
ngmodel.ngram_prob_[corp.nlp.vocab.strings['Sam'], (corp.nlp.vocab.strings['am'], )]

ngmodel.ngram_prob_[corp.nlp.vocab.strings['do'], (SENT_START, corp.nlp.vocab.strings['I'])]

#ng_docs = ngrams(corp, n, tokens_as_hashes=True, sentences=True, join=False)
#ng_docs = ngrams_augment_sents(ng_docs, n)

