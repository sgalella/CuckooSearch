import unittest

import cuckoo_search as cs

limits = [-5, 5, -3, 3]
resolution = 100
num_iterations = 1000
num_individuals = 20
alpha = 0.5


def run_search(algorithm, num_iterations=100):
    for _ in range(num_iterations):
        algorithm.update()


class TestConvergence(unittest.TestCase):

    def test_sphere(self):
        landscape = cs.SphereLandscape(limits, resolution)
        search = cs.CuckooSearch(num_individuals, landscape, alpha=alpha)
        run_search(search, num_iterations)
        self.assertAlmostEqual(search.best_fitness, 1)

    def test_grickwank(self):
        landscape = cs.GrickwankLandscape(limits, resolution)
        search = cs.CuckooSearch(num_individuals, landscape, alpha=alpha)
        run_search(search, num_iterations)
        self.assertAlmostEqual(search.best_fitness, 1)

    def test_himmelblau(self):
        landscape = cs.HimmelblauLandscape(limits, resolution)
        search = cs.CuckooSearch(num_individuals, landscape, alpha=alpha)
        run_search(search, num_iterations)
        self.assertAlmostEqual(search.best_fitness, 1)

    def test_ackley(self):
        landscape = cs.AckleyLandscape(limits, resolution)
        search = cs.CuckooSearch(num_individuals, landscape, alpha=alpha)
        run_search(search, num_iterations)
        self.assertAlmostEqual(search.best_fitness, 1)

    def test_rastringin(self):
        landscape = cs.RastringinLandscape(limits, resolution)
        search = cs.CuckooSearch(num_individuals, landscape, alpha=alpha)
        run_search(search, num_iterations)
        self.assertAlmostEqual(search.best_fitness, 1)


if __name__ == '__main__':
    unittest.main()