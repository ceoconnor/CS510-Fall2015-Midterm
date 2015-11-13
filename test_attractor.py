from attractor import Attractor
from random import uniform, randint
from math import sqrt
from nose import with_setup

class test_attr():
    
    def init_test(self):
        """Tests the initial setup for the problem to ensure the initial values are set correctly, the increment exists, and there are empty arrays for our variables to be filled later"""
        a = Attractor()
        assert a.params[0] == 10.0, "\nError in assigning initial parameter s"
        assert a.params[1] == 28.0, "\nError in assigning initial parameter p"
        assert a.params[2] == 8.0/3.0, "\nError in assigning initial parameter b"
        assert len(a.params) > 0, "\nError in assigning initial parameters to params array"
        assert a.dt > 0, "\nError in setting up dt"
        
        
    def euler_test(self):
        """Tests Euler's method in Attractor and ensures the arrays are being filled to the correct and same sizes"""
        a = Attractor()
        a.evolve([1, 2, 3], 1)
        assert len(a.solution['x']) > 0, "\nError in Euler's solution for x"
        assert len(a.solution['y']) > 0, "\nError in Euler's solution for y"
        assert len(a.solution['z']) > 0, "\nError in Euler's solution for z"
        assert len(a.solution['x']) == len(a.solution['y']), "\nError in assigning values to arrays for solution x and/or y"
        assert len(a.solution['y']) == len(a.solution['z']), "\nError in assigning values to arrays for solution y and/or z"
        
    def rk2_test(self):
        """Tests the second order Runge Kutta method and ensures the arrays are being filled correctly and similarily"""
        a = Attractor()
        a.evolve([1, 2, 3], 2)
        assert len(a.solution['x']) > 0, "\nError in RK2 solution for x"
        assert len(a.solution['y']) > 0, "\nError in RK2 solution for y"
        assert len(a.solution['z']) > 0, "\nError in RK2 solution for z"
        assert len(a.solution['x']) == len(a.solution['y']), "\nError in assigning values to arrays for solution x and/or y"
        assert len(a.solution['y']) == len(a.solution['z']), "\nError in assigning values to arrays for solution y and/or z"
    
    
    def rk4_test(self):
        """Tests the fourth order Runge Kutta method and ensures the arrays are being filled correctly and similarily"""
        a = Attractor()
        a.evolve([1, 2, 3], 4)
        assert len(a.solution['x']) > 0, "\nError in RK4 solution for x"
        assert len(a.solution['y']) > 0, "\nError in RK4 solution for y"
        assert len(a.solution['z']) > 0, "\nError in RK4 solution for z"
        assert len(a.solution['x']) == len(a.solution['y']), "\nError in assigning values to arrays for solution x and/or y"
        assert len(a.solution['y']) == len(a.solution['z']), "\nError in assigning values to arrays for solution y and/or z"
        
    def save_test(self):
        """Tests to see if a save file was created by creating another save file"""
        a = Attractor()
        a.evolve()
        a.save()
