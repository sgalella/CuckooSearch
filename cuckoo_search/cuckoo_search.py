import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import levy


class CuckooSearch:
    """ Implements the Cuckoo Search algorithm proposed in Cuckoo Search via Lévy Flights. """
    def __init__(self, num_individuals, landscape, alpha=1, c=1.5):
        """
        Initialization of the main parameters of the algorithm.

        Args:
            num_individuals (int): Number of nests.
            landscape (FitnessLandscape): Fitness landscape to evaluate the cuckoo search.
            alpha (int, optional): Parameter to scale the step size. Defaults to 1.
            c (float, optional): Controls the tail of the lévy distribution. Set between 1 and 3. Defaults to 1.5.
        """
        self.alpha = alpha
        self.num_individuals = num_individuals
        self.landscape = landscape
        self.population = self._initialize_population()
        self.best_fitness = -np.inf
        self.levy = levy(scale=c)

    def _initialize_population(self):
        """
        Initializes each nest in a random position in the landscape.

        Returns:
            np.array: Coordinates for each nest.
        """
        population = np.random.random((self.num_individuals, 2))
        population[:, 0] = (self.landscape.limits[1] - self.landscape.limits[0]) * population[:, 0] + self.landscape.limits[0]
        population[:, 1] = (self.landscape.limits[3] - self.landscape.limits[2]) * population[:, 1] + self.landscape.limits[2]
        return population

    def _levy_flight(self, ind1):
        """
        Computes the levy flight for an individual.

        Args:
            ind1 (int): Index of the selected individual to move.

        Returns:
            tuple: Updated position after the lévy flight.
        """
        angle = 2 * np.pi * np.random.random()
        move = self.alpha * np.multiply(self.levy.rvs(), [np.cos(angle), np.sin(angle)])
        position = self.population[ind1, :] + move
        while (position[0] < self.landscape.limits[0] or position[0] > self.landscape.limits[1]
                or position[1] < self.landscape.limits[2] or position[1] > self.landscape.limits[3]):
            angle = 2 * np.pi * np.random.random()
            move = self.alpha * np.multiply(self.levy.rvs(), [np.cos(angle), np.sin(angle)])
            position = self.population[ind1, :] + move
        return position

    def update(self):
        """ Runs one iteration of the algorithm. """
        ind1, ind2 = np.random.choice(self.num_individuals, size=(2, ), replace=False)
        position = self._levy_flight(ind1)
        fitness_ind1 = self.landscape.evaluate_fitness(position)
        fitness_ind2 = self.landscape.evaluate_fitness(self.population[ind2, :])
        if fitness_ind1 > fitness_ind2:
            self.population[ind2, :] = position
        self.best_fitness = max(fitness_ind1, self.best_fitness)


def visualize_all(algorithm, num_iterations):
    """
     Shows the evolution of the solutions in the landscape.

    Args:
        algorithm (CuckooSearch): Cuckoo search initialized.
        num_iterations (int): Number of iterations to run the algorithm.
    """
    plt.figure(figsize=(8, 5))
    algorithm.landscape.plot()
    plt.colorbar(shrink=0.75)
    plt.ion()
    for _ in range(num_iterations):
        algorithm.update()
        plt.cla()
        algorithm.landscape.plot()
        plt.scatter(algorithm.population[:, 0], algorithm.population[:, 1], color='r', marker='*')
        plt.title(f'Best fitness: {algorithm.best_fitness:.4f}')
        plt.draw()
        plt.pause(0.01)
    plt.ioff()
    plt.show()


def visualize_end(algorithm, num_iterations):
    """
    Shows the last evolution of the solutions in the landscape.

    Args:
        algorithm (CuckooSearch): Cuckoo search initialized.
        num_iterations (int): Number of iterations to run the algorithm.
    """
    for _ in range(num_iterations):
        algorithm.update()
    plt.figure(figsize=(8, 5))
    algorithm.landscape.plot()
    plt.colorbar(shrink=0.75)
    plt.scatter(algorithm.population[:, 0], algorithm.population[:, 1], color='r', marker='*')
    plt.title(f'Best fitness: {algorithm.best_fitness:.4f}')
    plt.show()


def no_visualization(algorithm, num_iterations):
    """
    Runs the algorithm without visualization. Prints the best solution obtained.

    Args:
        algorithm (CuckooSearch): Cuckoo search initialized.
        num_iterations (int): Number of iterations to run the algorithm.
    """
    for _ in range(num_iterations):
        algorithm.update()
    print(algorithm.best_fitness)


def main():
    np.random.seed(1234)
    resolution = 100
    limits = [-5, 5, -3, 3]  # x_min, x_max, y_min, y_max
    num_iterations = 200
    landscape = opt.RastringinLandscape(limits, resolution)
    search = CuckooSearch(10, landscape, alpha=0.5)
    visualize_all(search, num_iterations)


if __name__ == '__main__':
    import optimization_functions as opt
    main()
