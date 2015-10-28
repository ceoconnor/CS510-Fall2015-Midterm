import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

class Attractor(object):
    
    def __init__(self, s=10.0, p=28.0, b=(8.0/3.0), start=0.0, end=80.0, points=10000):
        self.s = s
        self.p = p
        self.b = b
        self.params = np.array([self.s, self.p, self.b])
        self.start = start
        self.end = end
        self.points = points
        self.t = np.linspace(self.start, self.end, self.points)
        self.dt = (self.end - self.start)/self.points
    
    def euler(self, a = np.array([]), dt = 0):
        if dt == 0:
            dt = self.dt
        x = a[0]
        y = a[1]
        z = a[2]
        dx = self.params[0] * (y - x)
        dy = x * (self.params[1] - z) - y
        dz = (x * y) - (self.params[2] * z)
        xdt = x + (dx * self.dt)
        ydt = y + (dy * self.dt)
        zdt = z + (dz * self.dt)
        return np.array([xdt, ydt, zdt])
        
    def rk2(self, b = np.array([])):
        dt = self.dt / 2.0
        rk1 = self.euler(b)
        x1 = b[0] + rk1[0] * new_dt
        y1 = b[1] + rk1[1] * new_dt
        z1 = b[2] + rk1[2] * new_dt
        return self.euler(np.array([x1, y1, z1]), dt)
        
              
    def rk3(self, c = np.array([])):
        dt = self.dt / 2.0
        rk2 = self.rk1(c)
        x2 = c[0] + rk2[0] * new_dt
        y2 = c[1] + rk2[1] * new_dt
        z2 = c[2] + rk2[2] * new_dt
        return self.euler(np.array([x2, y2, z2]), dt)
        
    def rk4(self, d = np.array([])):
        dt = self.dt
        dt = self.dt
        dt = self.dt
        rk3 = self.rk2(d)
        x3 = d[0] + rk3[0] * self.dt
        y3 = d[1] + rk3[1] * self.dt
        z3 = d[2] + rk3[2] * self.dt
        return self.euler(np.array([x3, y3, z3]), dt)
        
                            
    def evolve(self, r0 = np.array([0.1, 0.0, 0.0]), order=1):
        if order == 1:
            sol = self.euler(r0)
        elif order == 2:
            sol = self.rk2(r0)
        elif order == 4:
            sol = self.rk4(r0)
        else:
            return 'Uh oh'
        
        self.solution = pd.DataFrame(sol, columns=['t','x','y','z'])
        return self.solution
        
    def save(self):
        self.solution.to_csv("export.csv")
        
    def plotx(self):
        plt.plot(self.solution['x'], color='k')
        plt.title("Plot tx")
        plt.tlabel("t")
        plt.xlabel("X")
        
        
    def ploty(self):
        plt.plot(self.solution['y'], color='k')
        plt.title("Plot ty")
        plt.tlabel("t")
        plt.ylabel("Y")
        
    def plotz(self):
        plt.plot(self.solution['z'], color='k')
        plt.title("Plot tz")
        plt.tlabel("t")
        plt.zlabel("Z")
        
    def plotxy(self):
        plt.plot(self.solution['x'], self.solution['y'], color='k')
        plt.title("Plot xy")
        plt.xlabel("X")
        plt.ylabel("Y")
        
    def plotyz(self):
        plt.plot(self.solution['y'], self.solution['z'], color='k')
        plt.title("Plot yz")
        plt.ylabel("Y")
        plt.zlabel("Z")
        
    def plotzx(self):
        plt.plot(self.solution['z'], self.solution['x'], color='k')
        plt.title("Plot zx")
        plt.zlabel("Z")
        plt.xlabel("X")
    
    def plot3d(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.plot(self.solution['x'], self.solution['y'], self.solution['z'])
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Lorenz Attractor")

        plt.show()
