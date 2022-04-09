import matplotlib.pyplot as plt
import numpy as np


class EnvFunctions:
    def __init__(self):
        self.figure_count = 0

    def plot_array(self, array: list, name: str, x_label: str = None, y_label: str = None, additional_infos: list = None):
        plt.figure(self.figure_count)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        fig_name = name
        if additional_infos is not None:
            for info in additional_infos:
                fig_name += info
                fig_name += "__"
        plt.plot(array)
        plt.savefig('Figures/{}-{}.png'.format(fig_name, self.figure_count))
        self.figure_count += 1



