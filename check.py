# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:23:46 2020

@author: ALIV
"""

'''import random
l=random.sample(range(100), 10)
print(l)
l=random.sample(range(0, 10), 10)
print(l)
print( random.randint(5, 10))
p1, p2 = [random.randrange(1, 50) for i in range(2)]
print(p1,p2)'''
import random
import numpy as np 

route=random.sample(range(10),10)
print(route)
newr=route[:]
print(newr)
newr[2:5]=route[1:4:-1]
print(newr)
#==================
def cost(cost_mat, route):
   return cost_mat[np.roll(route, 1), route].sum()

cost_mat = np.array([
   [0, 1, 2, 3],
   [1, 0, 4, 5],
   [2, 4, 0, 7],
   [3, 5, 7, 0],
])

route = np.array([2, 1, 3, 0])

print(cost(cost_mat, route))