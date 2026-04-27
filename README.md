# independent-study
CS236 Final Project README
Authors: Kayla Mullen, Ela Ben Saad, Elisha Andrews
Date: Dec 18, 2025

BaselineBellmanFord.py can be run in terminal as usual. 
    This will output baseline results for testing different dasher amounts.
    There are additional test blocks above this line that are currently commented out. 
    Each test block reads in tasks and dashers, creates a new simulator, runs it and prints the results.

    get_results will perform the above block 10x and output the results of each.

    packages
        - heapq
        - pandas
        - scipy
        - numpy


from __future__ import annotations
import heapq
import itertools
from typing import Any, Optional, Tuple
from numpy import random
import math
from WeightedGraph import WeightedGraph
import pandas as pd
from scipy.sparse import csr_array
from scipy.sparse.csgraph import bellman_ford


SmartDispatch can be run in terminal as usual.
    This will output SmartDispatch results for testing different dasher amounts.
    There are additional test blocks above this line that are currently commented out. 
    Each test block reads in tasks and dashers, creates a new simulator, runs it and prints the results.

    get_results will perform the above block 10x and output the results of each.

    packages
        - heapq
        - numpy
        - pandas
        - scipy
        - networkx


Predict.py can be run in terminal as usual.
    predict_reward_and_min(tasks,45,1810) will take in tasks and predict when the next one will occur and its reward
    test_performance performs a cross validation test on all vertices as described in the evaluation section of the paper

    packages:
        - pandas
        - prophet
        - random
        - sklearn