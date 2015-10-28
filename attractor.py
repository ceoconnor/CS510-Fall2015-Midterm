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
        self.dt = (self.end - self.start)/self.points
        self.t = np.arange(self.start+1, self.end, self.dt)
    
    def euler(self, a = np.array([]), dt = 0):
        if dt == 0:
            dt = self.dt
        x = a[0]
        y = a[1]
        z = a[2]
        dx = self.params[0] * (y - x)
        dy = x * (self.params[1] - z) - y
        dz = (x * y) - (self.params[2] * z)
        dtx = x + (dx * self.dt)
        dty = y + (dy * self.dt)
        dtz = z + (dz * self.dt)
        return np.array([dtx, dty, dtz])
        
    def rk2(self, b = np.array([])):
        dt = self.dt / 2.0
        rk1 = self.euler(b)
        x1 = b[0] + rk1[0] * dt
        y1 = b[1] + rk1[1] * dt
        z1 = b[2] + rk1[2] * dt
        return self.euler(np.array([x1, y1, z1]), dt)
        
              
    def rk3(self, c = np.array([])):
        dt = self.dt / 2.0
        rk2 = self.rk1(c)
        x2 = c[0] + rk2[0] * dt
        y2 = c[1] + rk2[1] * dt
        z2 = c[2] + rk2[2] * dt
        return self.euler(np.array([x2, y2, z2]), dt)
        
    def rk4(self, d = np.array([])):
        dt = self.dt
        rk3 = self.rk2(d)
        x3 = d[0] + rk3[0] * dt
        y3 = d[1] + rk3[1] * dt
        z3 = d[2] + rk3[2] * dt
        return self.euler(np.array([x3, y3, z3]), dt)
        
                            
    def evolve(self, r0 = np.array([0.1, 0.0, 0.0]), order=4):
        ts = np.append(self.t, self.end)
        sol = np.array(np.append(0, r0))
        
        if order == 1:    
            for i in ts:
                e = self.euler(r0)
                sol = np.vstack((sol,np.append(i, e)))
        elif order == 2:
            for i in ts:
                e = self.rk2(r0)
                sol = np.vstack((sol,np.append(i, e)))
        elif order == 4:
            for i in ts:
                e = self.rk4(r0)
                sol = np.vstack((sol,np.append(i, e)))
        else:
            return 'Uh oh'
        
        self.solution = pd.DataFrame(sol)
        self.solution.columns =['t','x','y','z']
        return self.solution
        
    def save(self):
        self.solution.to_csv("solution.csv")
        
    def plotx(self):
        plt.plot(self.solution['x'])
        plt.show()
        
        
    def ploty(self):
        plt.plot(self.solution['y'])
        plt.show()
        
    def plotz(self):
        plt.plot(self.solution['z'])
        plt.show()
        
    def plotxy(self):
        plt.plot(self.solution['t'], self.solution['x'], 'k')
        plt.plot(self.solution['t'], self.solution['y'], 'r')
        plt.show()
        
    def plotyz(self):
        plt.plot(self.solution['t'], self.solution['y'], 'k')
        plt.plot(self.solution['t'], self.solution['z'], 'r')
        plt.show()
        
    def plotzx(self):
        plt.plot(self.solution['t'], self.solution['z'], 'k')
        plt.plot(self.solution['t'], self.solution['x'], 'r')
        plt.show()
    
    def plot3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.solution['x'], self.solution['y'], self.solution['z'])
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Lorenz Attractor")

        plt.show()
