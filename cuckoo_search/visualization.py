import matplotlib.pyplot as plt


class VisualizeSearch:
    @staticmethod
    def show_all(algorithm, num_iterations=100):
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

    @staticmethod
    def show_last(algorithm, num_iterations=100):
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
