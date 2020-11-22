import numpy as np

import optimization_functions as opt
from visualization import VisualizeSearch
from cuckoo_search import CuckooSearch


np.random.seed(1234)
resolution = 100
limits = [-5, 5, -3, 3]  # x_min, x_max, y_min, y_max
num_iterations = 200
landscape = opt.SphereLandscape(limits, resolution)
search = CuckooSearch(10, landscape, alpha=0.5)
VisualizeSearch.show_last(search, num_iterations)
