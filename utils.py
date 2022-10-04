"""
Part of the code in <utils.py> is from SimGNN@benedekrozemberczki
"""

import os
import numpy as np
from texttable import Texttable
from scipy import stats


def computing_precision_ks(trues, predictions, ks, inc