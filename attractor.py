import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


class Attractor(object):
    
    def __init__(self, s=10.0, p=28.0, b=(8.0/3.0), start=0.0, end=80.0, points=10000):
        """Defines the initial values and sets up empty numpy arrays for x, y, z, and t to be able to fill later"""
        """Creates the step (dt) that will be used to increment the iterator through the methods: Euler, rk2, and rk4"""
        self.s = s
        self.p = p
        self.b = b
        self.params = np.array([self.s, self.p, self.b])
        self.start = start
        self.end = end
        self.points = points
        self.dt = (self.end - self.start)/self.points
        self.x = np.zeros([])
        self.y = np.zeros([])
        self.z = np.zeros([])
        self.t = np.zeros([])
    
    def fx (self, xt, yt, sx, dxt):
        """Define dx(t)/dt = s[y(t) - x(t)] to be used in the methods: Euler, rk2, rk4"""
        return sx*(yt - xt)*dxt
    
    def fy (self, xt, yt, zt, py, dyt):
        """Define dy(t)/dt = x(t)[p - z(t)] - y(t) to be used in the methods: Euler, rk2, rk4"""
        return (xt*(py - zt) - yt)*dyt
    
    def fz (self, xt, yt, zt, bz, dzt):
        """Define dz(t)/dt = x(t) * y(t) - b * z(t) to be used in the methods: Euler, rk2, rk4"""
        return (xt*yt - bz*zt)*dzt
     
    def euler(self, a):
        """Creates numpy arrays to be filled through the iteration of Euler's numerical approximation method for each variable (x, y, z) and function (fx, fy, fz)"""
        self.x = np.zeros(self.points + 1)
        self.y = np.zeros(self.points + 1)
        self.z = np.zeros(self.points + 1)
        self.t = np.zeros(self.points + 1)
        
        self.x[0] = a[0]
        self.y[0] = a[1]
        self.z[0] = a[2]
        
        for i in xrange(self.points):
            self.x[i+1] = self.x[i] + self.fx(self.x[i], self.y[i], self.s, self.dt)
            self.y[i+1] = self.y[i] + self.fy(self.x[i], self.y[i], self.z[i], self.p, self.dt)
            self.z[i+1] = self.z[i] + self.fz(self.x[i], self.y[i], self.z[i], self.b, self.dt)
            self.t[i+1] = self.t[i] + self.dt
        
        """Returns Euler's solution to the functions: fx, fy, and fz"""
        return (self.x, self.y, self.z)
        
    def rk2(self, a):
        """Applies the second order Runge Kutta to the numpy arrays x, y, z, and t"""
        self.x = np.zeros(self.points + 1)
        self.y = np.zeros(self.points + 1)
        self.z = np.zeros(self.points + 1)
        self.t = np.zeros(self.points + 1)
        
        self.x[0] = a[0]
        self.y[0] = a[1]
        self.z[0] = a[2]
        
        for i in xrange(self.points):
            k1x = self.fx(self.x[i], self.y[i], self.s, self.dt)
            k2x = self.fx(k1x*0.5*self.dt + self.x[i], self.y[i], self.s, self.dt)
            self.x[i+1] = self.x[i] + 0.5*(k1x + k2x)
            
            
            k1y = self.fy(self.x[i], self.y[i], self.z[i], self.p, self.dt)
            k2y = self.fy(k1y*0.5*self.dt + self.x[i], self.y[i], self.z[i], self.p, self.dt)
            self.y[i+1] = self.y[i] + 0.5*(k1y + k2y)
            
            k1z = self.fz(self.x[i], self.y[i], self.z[i], self.b, self.dt)
            k2z = self.fz(k1z*0.5*self.dt + self.x[i], self.y[i], self.z[i], self.b, self.dt)
            self.z[i+1] = self.z[i] + 0.5*(k1z + k2z)
            
            self.t[i+1] = self.t[i] + self.dt
        
        """Returns the second order Runge Kutta solution to the functions: fx, fy, and fz"""
        return (self.x, self.y, self.z)
        
              
    def rk4(self, a):
        """Apllies the third and fourth order Runga Kuttas to the numpy arrays x, y, z, and t"""
        self.x = np.zeros(self.points + 1)
        self.y = np.zeros(self.points + 1)
        self.z = np.zeros(self.points + 1)
        self.t = np.zeros(self.points + 1)
        
        self.x[0] = a[0]
        self.y[0] = a[1]
        self.z[0] = a[2]
        
        for i in xrange(self.points):
            k1x = self.fx(self.x[i], self.y[i], self.s, self.dt)
            k2x = self.fx(k1x*0.5*self.dt + self.x[i], self.y[i], self.s, self.dt)
            k3x = self.fx(k2x*0.5*self.dt + self.x[i], self.y[i], self.s, self.dt)
            k4x = self.fx(k3x*self.dt + self.x[i], self.y[i], self.s, self.dt)
            self.x[i+1] = self.x[i] + (1.0/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
            
            
            k1y = self.fy(self.x[i], self.y[i], self.z[i], self.p, self.dt)
            k2y = self.fy(k1y*0.5*self.dt + self.x[i], self.y[i], self.z[i], self.p, self.dt)
            k3y = self.fy(k2y*0.5*self.dt + self.x[i], self.y[i], self.z[i], self.p, self.dt)
            k4y = self.fy(k3y*self.dt + self.x[i], self.y[i], self.z[i], self.p, self.dt)
            self.y[i+1] = self.y[i] + (1.0/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
            
            k1z = self.fz(self.x[i], self.y[i], self.z[i], self.b, self.dt)
            k2z = self.fz(k1z*0.5*self.dt + self.x[i], self.y[i], self.z[i], self.b, self.dt)
            k3z = self.fz(k2z*0.5*self.dt + self.x[i], self.y[i], self.z[i], self.b, self.dt)
            k4z = self.fz(k3z*self.dt + self.x[i], self.y[i], self.z[i], self.b, self.dt)
            self.z[i+1] = self.z[i] + (1.0/6.0)*(k1z + 2*k2z + 2*k3z + k4z)
            
            self.t[i+1] = self.t[i] + self.dt
        
        """Returns the fourth order Runge Kutta solution to the functions: fx, fy, and fz"""
        return (self.x, self.y, self.z)
        
        
                            
    def evolve(self, r0=np.array([0.1,0.0,0.0]), order=4):
        """Defines the initial values for x, y, and z"""
        x0 = r0[0]
        y0 = r0[1]
        z0 = r0[2]
        
        """Determines which order of Runge Kutta to apply to x, y, and z"""
        if order == 1:    
            self.euler(r0)
        elif order == 2:
            self.rk2(r0)
        elif order == 4:
            self.rk4(r0)
        else:
            return 'Uh oh! Houston, we have a PROBLEM. Pleae choose 1, 2, or 4. There is no other option human.'
        
        """Returns the solution to the specified method and stores it into a pandas dataframe with labeled columns"""
        self.solution = pd.DataFrame({'t': self.t, 'x': self.x, 'y': self.y, 'z': self.z})
        return self.solution
        
    def save(self):
        """Saves the result to a csv file called 'solution.'"""
        self.solution.to_csv("solution.csv")
        
    def plotx(self):
        """Plots the values of x with respect to t"""
        plt.plot(self.solution['x'])
        plt.suptitle('X vs Time')
        plt.xlabel('t')
        plt.ylabel('x(t)')
        plt.show()
        
        
    def ploty(self):
        """Plots the values of y with respect to t"""
        plt.plot(self.solution['y'])
        plt.suptitle('Y vs Time')
        plt.xlabel('t')
        plt.ylabel('y(t)')
        plt.show()
        
    def plotz(self):
        """Plots the values of z with respect to t"""
        plt.plot(self.solution['z'])
        plt.suptitle('Z vs Time')
        plt.xlabel('t')
        plt.ylabel('z(t)')
        plt.show()
        
    def plotxy(self):
        """Plots the values of x and y with respect to t"""
        plt.plot(self.solution['x'], self.solution['y'], 'k')
        plt.suptitle('x(t) vs y(t)')
        plt.xlabel('x(t)')
        plt.ylabel('y(t)')
        plt.show()
        
    def plotyz(self):
        """Plots the values of y and z with respect to t"""
        plt.plot(self.solution['y'], self.solution['z'], 'b')
        plt.suptitle('y(t) vs z(t)')
        plt.xlabel('y(t)')
        plt.ylabel('z(t)')
        plt.show()
        
    def plotzx(self):
        """Plots the values of z and x with respect to t"""
        plt.plot(self.solution['z'], self.solution['x'], 'r')
        plt.suptitle('z(t) vs x(t)')
        plt.xlabel('z(t)')
        plt.ylabel('x(t)')
        plt.show()
    
    def plot3d(self):
        """Plots the values of x, y, and z with respect to t in a 3D graph"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.solution['x'], self.solution['y'], self.solution['z'])
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Lorenz Attractor")

        plt.show()
