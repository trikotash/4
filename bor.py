
import numpy as np
from Pola import Electromagneticfield, Partcile
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, coords, momentum):
        self.coords = coords
        self.momentum = momentum
        self.size = 100000
        self.time = np.linspace(0, 600, self.size)

    def coord(self, a, b):
        d = {1: 'x', 2: 'y', 3: 'z'}
        plt.plot(self.coords[:, a-1], self.coords[:, b-1])
        plt.xlabel(d[a])
        plt.ylabel(d[b])
        plt.grid()
        plt.show()

    def coord3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2], label='3Dtrajec')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def moment(self, b):
        d = {1: '$p_x$', 2: '$p_y$', 3: '$p_z$'}
        plt.plot(self.time, self.momentum[:, b-1])
        plt.xlabel('t')
        plt.ylabel(d[b])
        plt.grid()
        plt.show()



    def intens_c(self):
        k = Intensity(self.coords, self.momentum)
        g = k.intens()
        plt.plot(np.linspace(0, 3, 100), g)
        plt.grid()
        plt.show()

field = ElectromagneticField('wave', 1, [1, 0, 0])
par = Particle([0, 0, 0], [0.1, 0, 0], field)

solver = Boris(par, 1)

a, b = solver.evaluate(0)


kapa = Plotter(b, a)
kapa.intens_c()
