from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial


class FitnessLandscape(ABC):
    def __init__(self, limits, resolution):
        self.limits = limits
        self.resolution = resolution
        self.X, self.Y = self._create_meshgrid()
        self.coords, self.tree = self._generate_coords()
        self.fitness_function = self._calculate_fitness().reshape(self.resolution, self.resolution)

    def _generate_coords(self):
        coords = np.dstack([self.X.ravel(), self.Y.ravel()])[0]
        return coords, spatial.cKDTree(coords)

    def _create_meshgrid(self):
        x = np.linspace(self.limits[0], self.limits[1], self.resolution)
        y = np.linspace(self.limits[2], self.limits[3], self.resolution)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def evaluate_fitness(self, pos):
        _, index = self.tree.query(pos)
        return np.fabs(self.fitness_function[index // self.resolution][index % self.resolution] - np.max(self.fitness_function))

    def plot(self):
        cs = plt.contour(self.X, self.Y, self.fitness_function)
        plt.clabel(cs, inline=1, fontsize=6)
        plt.imshow(self.fitness_function, extent=self.limits, origin="lower", alpha=0.3)

    @abstractmethod
    def _calculate_fitness(self):
        pass


class SphereLandscape(FitnessLandscape):
    def _calculate_fitness(self):
        return self.X ** 2 + self.Y ** 2


class GrickwankLandscape(FitnessLandscape):
    def _calculate_fitness(self):
        return 1 + (self.X ** 2 / 4000) + (self.Y ** 2 / 4000) - np.cos(self.X / np.sqrt(2)) - np.cos(self.Y / np.sqrt(2))


class HimmelblauLandscape(FitnessLandscape):
    def _calculate_fitness(self):
        return (self.X ** 2 + self.Y - 11) ** 2 + (self.X + self.Y ** 2 - 7) ** 2


class AckleyLandscape(FitnessLandscape):
    def _calculate_fitness(self):
        return (-20 * np.exp(-0.2 * np.sqrt(0.5 * (self.X ** 2 + self.Y ** 2))) - np.exp(0.5 * np.cos(2 * np.pi * self.X)
                + np.cos(2 * np.pi * self.Y)) + np.exp(1) + 20)


class RastringinLandscape(FitnessLandscape):
    def _calculate_fitness(self):
        return 20 + self.X ** 2 - 10 * np.cos(2 * np.pi * self.X) - 10 * np.cos(2 * np.pi * self.Y)
