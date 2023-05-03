"""
Example script that shows how to fit an n-gram model with different built-in corpora and generate some sentences
given some seed input.

This examples requires that you have installed tmtoolkit with the recommended set of packages and have installed an
English and a German language model for spaCy:

    pip install -U "tmtoolkit[recommended]"
    python -m tmtoolkit setup en,de

.. codeauthor:: Markus Konrad <post@mkonrad.net>
.. date:: Nov. 2022
"""

import tmtoolkit.corpus as c
from tmtoolkit.ngrammodels import NGramModel

#%%

SEEDS = {
    'de-parlspeech-v2-sample-bundestag': {
        3: ['Vielen', 'Dank'],
        4: ['Ich', 'möchte', 'Ihnen'],
    },
    'en-parlspeech-v2-sample-houseofcommons': {
        3: ['Thank', 'you'],
        4: ['I', 'want', 'to'],
    }
}

#%%

for corp_name, n_seeds in SEEDS.items():
    print(f'loading and tokenizing corpus "{corp_name}"...')
    corp = c.Corpus.from_builtin_corpus(corp_name, max_workers=1.0)
    c.print_summary(corp)

    for n, seed in n_seeds.items():
        print(f'> fitting {n}-gram model...')
        ngmodel = NGramModel(n, tokens_as_hashes=False)
        ngmodel.fit(corp)

        seed_joined = ' '.join(seed)
        print(f'> generating some sequences with seed input "{seed_joined}"')
        for i in range(1, 11):
            s = seed_joined + ' ' + ' '.join(ngmodel.generate_sequence(seed))
            print(f'>> generated sample #{i}: {s}')

        print('\n')
    print('---\n')

#%%

print('done.')