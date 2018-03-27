'''
Created on 11 Mar 2018

@author: meierfra
'''

import matplotlib.pyplot as plt


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
