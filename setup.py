"""
tmtoolkit setuptools based setup module

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""

import os
import sys
from codecs import open

from setuptools import setup, find_packages

__title__ = 'tmtoolkit'
__version__ = '0.12.0'
__author__ = 'Markus Konrad'
__license__ = 'Apache License 2.0'


GITHUB_URL = 'https://github.com/internaut/tmtoolkit'

DEPS_BASE = ['numpy>=1.23.0,<2.0', 'scipy>=1.7.0,<2.0', 'globre>=0.1.5,<0.2',
             'pandas>=1.4.0,<3.0', 'xlrd>=2.0.0', 'openpyxl>=3.0.0,<4.0',
             'matplotlib>=3.5.0,<4.0', 'bidict>=0.21.0,<1.0', 'wheel>=0.40,<1.0']

DEPS_EXTRA = {
    'textproc': ['spacy>=3.3.0,<4.0', 'loky>=3.0.0,<4.0'],
    'textproc_extra': ['PyICU>=2.8,<3.0', 'nltk>=3.6.0,<3.9'],
    'wordclouds': ['wordcloud>=1.8.0,<2.0', 'Pillow>=9.0.0,<10.0.0'],
    'lda': ['lda>=2.0,<3.0'],
    'sklearn': ['scikit-learn>=1.0.0,<2.0'],
    'gensim': ['gensim>=4.1.0,<5.0'],
    'topic_modeling_eval_extra': ['gmpy2>=2.1.0,<3.0'],
    'rinterop': ['rpy2>=3.5.0,<4.0'],
    'test': ['pytest>=7.2.0,<8.0', 'hypothesis>=6.49.0,<7.0'],
    'doc': ['Sphinx>=6.1.0,<7.0', 'sphinx-rtd-theme>=1.2.0', 'nbsphinx>=0.9.0,<1.0'],
    'dev': ['coverage>=7.2,<8.0', 'coverage-badge>=1.1.0', 'pytest-cov>=4.0.0,<5.0', 'twine>=4.0',
            'ipython>=8.12.0', 'jupyter>=1.0.0', 'notebook>=6.5.0', 'tox>=4.4,<5.0', 'setuptools>=67.6'],
}

# DEPS_EXTRA['minimal'] = DEPS_BASE   # doesn't work with extras_require and pip currently
# see https://github.com/pypa/setuptools/issues/1139

if sys.version_info >= (3, 11):
    print('warning: packages "wordcloud" and "lda" are not available for Python 3.11 and above at the moment',
          file=sys.stderr)
    DEPS_EXTRA['wordclouds'] = []
    DEPS_EXTRA['lda'] = []

DEPS_EXTRA['recommended'] = DEPS_EXTRA['textproc'] + DEPS_EXTRA['wordclouds']
DEPS_EXTRA['all'] = []
for k, deps in DEPS_EXTRA.items():
    if k not in {'recommended', 'all'}:
        DEPS_EXTRA['all'].extend(deps)

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=__title__,
    version=__version__,
    description='Text Mining and Topic Modeling Toolkit',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url=GITHUB_URL,
    project_urls={
        'Bug Reports': GITHUB_URL + '/issues',
        'Source': GITHUB_URL,
    },

    author=__author__,
    author_email='post@mkonrad.net',

    license=__license__,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',

        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',

        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],

    keywords='textmining textanalysis text mining analysis preprocessing topicmodeling topic modeling evaluation',

    packages=find_packages(exclude=['tests', 'examples']),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=DEPS_BASE,
    extras_require=DEPS_EXTRA
)
