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
        self.t = np.linspace(start,end,points)
        self.dt = np.abs(self.t[1]-self.t[0])
    
    def deriv(self):
        self.x = np.empty([self.points + 1.0])
        self.y = np.empty([self.points + 1.0])
        self.z = np.empty([self.points + 1.0])
        self.dx = self.s * (self.y - self.x)
        self.dy = self.x * (self.p - self.z)  - self.y
        self.dz = self.x * self.y - self.b * self.z
        return self.dx, self.dy, self.dz
    
    def euler(self):
        self.k1x = np.array([])
        self.k1y = np.array([])
        self.k1z = np.array([])
        for i in np.arange(len(self.t)):
            self.k1x = np.append(self.k1x, self.dx[i]*self.x*self.dt)
            self.k1y = np.append(self.k1y, self.dy[i]*self.y*self.dt)
            self.k1z = np.append(self.k1z, self.dz[i]*self.z*self.dt)
            self.x[i] = self.x[i] + self.k1x[i]
            self.y[i] = self.y[i] + self.k1y[i]
            self.z[i] = self.z[i] + self.k1z[i]
        return self.x, self.y, self.z
        
    def rk2(self):
        self.k1x = np.array([])
        self.k1y = np.array([])
        self.k1z = np.array([])
        self.k2x = np.array([])
        self.k2y = np.array([])
        self.k2z = np.array([])
        self.xk = np.array([])
        self.yk = np.array([])
        self.zk = np.array([])
        for i in np.arange(len(self.t)):
            self.k1x = np.append(self.k1x, self.dx[i]*self.x*self.dt)
            self.k1y = np.append(self.k1y, self.dy[i]*self.y*self.dt)
            self.k1z = np.append(self.k1z, self.dz[i]*self.z*self.dt)
            self.xk = np.append(self.xk, self.x[i] + self.k1x[i] * 0.5)
            self.yk = np.append(self.yk, self.y[i] + self.k1y[i] * 0.5)
            self.zk = np.append(self.zk, self.z[i] + self.k1z[i] * 0.5)
            self.k2x = np.append(self.k2x, self.dx[i] * self.xk * self.dt)
            self.k2y = np.append(self.k2y, self.dy[i] * self.yk * self.dt)
            self.k2z = np.append(self.k2z, self.dz[i] * self.zk * self.dt)
            self.x[i] = self.x[i] + (0.5 * self.k2x[i])
            self.y[i] = self.y[i] + (0.5 * self.k2y[i])
            self.z[i] = self.z[i] + (0.5 * self.k2z[i])
        return self.x, self.y, self.z
              
    def rk4(self):
        self.k1x = np.array([])
        self.k1y = np.array([])
        self.k1z = np.array([])
        self.k2x = np.array([])
        self.k2y = np.array([])
        self.k2z = np.array([])
        self.k3x = np.array([])
        self.k3y = np.array([])
        self.k3z = np.array([])
        self.k4x = np.array([])
        self.k4y = np.array([])
        self.k4z = np.array([])
        self.xk = np.array([])
        self.yk = np.array([])
        self.zk = np.array([])
        
        for i in np.arange(len(self.t)):
            self.k1x = np.append(self.k1x, self.dx[i]*self.x*self.dt)
            self.k1y = np.append(self.k1y, self.dy[i]*self.y*self.dt)
            self.k1z = np.append(self.k1z, self.dz[i]*self.z*self.dt)
            self.xk = np.append(self.xk, self.x[i] + self.k1x[i] * 0.5)
            self.yk = np.append(self.yk, self.y[i] + self.k1y[i] * 0.5)
            self.zk = np.append(self.zk, self.z[i] + self.k1z[i] * 0.5)
            self.k2x = np.append(self.k2x, self.dx[i] * self.xk * self.dt)
            self.k2y = np.append(self.k2y, self.dy[i] * self.yk * self.dt)
            self.k2z = np.append(self.k2z, self.dz[i] * self.zk * self.dt)
            self.xk[i] = self.x[i] + self.k2x[i] * 0.5
            self.yk[i] = self.y[i] + self.k2y[i] * 0.5
            self.zk[i] = self.z[i] + self.k2z[i] * 0.5
            self.k3x = np.append(self.k3x, self.dx[i] * self.xk * self.dt)
            self.k3y = np.append(self.k3y, self.dy[i] * self.yk * self.dt)
            self.k3z = np.append(self.k3z, self.dz[i] * self.zk * self.dt)
            self.xk[i] = self.x[i] + self.k3x[i]
            self.yk[i] = self.y[i] + self.k3y[i]
            self.zk[i] = self.z[i] + self.k3z[i]
            self.k4x = np.append(self.k4x, self.dx[i] * self.xk * self.dt)
            self.k4y = np.append(self.k4y, self.dy[i] * self.yk * self.dt)
            self.k4z = np.append(self.k4z, self.dz[i] * self.zk * self.dt)
            self.x[i] = self.x[i] + (self.k1x[i] + 2*(self.k2x[i] + self.k3x[i]) + self.k4x[i])/6.0
            self.y[i] = self.y[i] + (self.k1y[i] + 2*(self.k2y[i] + self.k3y[i]) + self.k4y[i])/6.0
            self.z[i] = self.z[i] + (self.k1z[i] + 2*(self.k2z[i] + self.k3z[i]) + self.k4z[i])/6.0
        return self.x, self.y, self.z
                            
    def evolve(self, x0=0.1, y0=0.0, z0=0.0, order=4):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.x[0] = self.x0
        self.y[0] = self.y0
        self.z[0] = self.z0
        self.r0=np.array([self.x0, self.y0, self.z0])
        
        if order == 1:
            self.solution = pd.DataFrame(self.euler())
        elif order == 2:
            self.solution = pd.DataFrame(self.rk2())
        elif order == 4:
            self.solution = pd.DataFrame(self.rk4())
        else:
            self.solution = 0
        return self.solution
        
    def save(self):
        self.solution.to_csv("export.csv")
        
    def plotx(self):
        plt.plot(self.t, self.x, color='k')
        plt.title("Plot tx")
        plt.tlabel("t")
        plt.xlabel("X")
        
        
    def ploty(self):
        plt.plot(self.t, self.y, color='k')
        plt.title("Plot ty")
        plt.tlabel("t")
        plt.ylabel("Y")
        
    def plotz(self):
        plt.plot(self.t, self.z, color='k')
        plt.title("Plot tz")
        plt.tlabel("t")
        plt.zlabel("Z")
        
    def plotxy(self):
        plt.plot(self.x, self.y, color='k')
        plt.title("Plot xy")
        plt.xlabel("X")
        plt.ylabel("Y")
        
    def plotyz(self):
        plt.plot(self.y, self.z, color='k')
        plt.title("Plot yz")
        plt.ylabel("Y")
        plt.zlabel("Z")
        
    def plotzx(self):
        plt.plot(self.z, self.x, color='k')
        plt.title("Plot zx")
        plt.zlabel("Z")
        plt.xlabel("X")
    
    def plot3d(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.plot(self.x, self.y, self.z)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Lorenz Attractor")

        plt.show()
