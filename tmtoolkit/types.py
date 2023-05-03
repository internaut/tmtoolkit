"""
Module with common types used in type annotations throughout this project.

.. codeauthor:: Markus Konrad <post@mkonrad.net>
"""

from enum import IntEnum
from typing import Union


Proportion = IntEnum('Proportion', 'NO YES LOG', start=0)

StrOrInt = Union[str, int]
