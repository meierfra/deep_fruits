'''
Created on 11 Mar 2018

@author: meierfra
'''

import matplotlib.pyplot as plt
import numpy as np
import time


class live_plot():
    plotlist = []
    fig = None
    ax = None
    data_set = []

    def __init__(self, lables=None):
        nplots = len(lables) if lables else 1
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)
        self.fig.show()
        for i in range(nplots):
            self.data_set.append([])
            p, = self.ax.plot([], [], '-', label=lables[i])
            self.plotlist.append(p)
        self.ax.legend()

    def _draw(self):
        plt.pause(0.0001)

    def update_point(self, data_point, plot_idx=0, draw=True):
        data = self.data_set[plot_idx]
        data.append(data_point)
        self.plotlist[plot_idx].set_data(range(0, len(data)), data)
        self.ax.relim()
        self.ax.autoscale_view()
        if draw:
            self._draw()

    def update_points(self, data_points=[], draw=True):
        for i in range(len(self.data_set)):
            self.update_point(data_points[i], plot_idx=i, draw=False)
        if draw:
            self._draw()

    def save(self, filename):
        self.fig.savefig(filename)


if __name__ == "__main__":
    lp = live_plot(["plot1", "plot2"])

    for i in range(0, 100):
        dp1 = np.sin(i * 2 * np.pi / 100)
        dp2 = np.sin(i * 2 * np.pi / 30)
        lp.update_points([dp1, dp2])
        if (i + 1) % 25 == 0:
            lp.save('live_plot_fig_' + str(i + 1) + '.png')

        time.sleep(0.1)
    # plt.show()
