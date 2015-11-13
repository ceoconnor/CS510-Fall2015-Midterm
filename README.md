# CS510-Fall2015-Midterm
Created by Cormac O'Connor



This repository contains the files for the CS510 midterm. The files included are: attractor.py, test_attractor.py, and explore_attractor.py.

attractor.py contains the source code for a Lorenz Attractor. It is written in Python and uses classes to implement the fourth order Runga Kutta for the Lorenz differential equations.
The code creates arrays of each step of the numerical apporximation and then graphs it in 1D, 2D, and 3D plots.

test_attractor.py contains the source code for tests to run on attractor.py's various methods to ensure they return the correct result. It contains a test for Euler, RK2, RK3, RK4, and evolve methods.

explore_attractor.py contains an iPython Jupiter notebook that shows the graphical representations of the Lorenz Attractor and its components broken up (1D and 2D plots). It also shows how sensitive the functions are to the initial conditions and shows various outcomes from different initial values.

