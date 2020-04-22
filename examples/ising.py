#!/bin/python3

import numpy as np
import draw_ising 


### Set up parameters
L = 128
T = 1.0
updates_per_frame = 10000

# Start with all spins 1
spin = np.ones([L,L], dtype=int) 



################################################
# Assing the increase T and decrease T buttons #
def tup(event):
   global T
   print("up")
   T = T+0.1

def tdn(event):
   global T
   print("down")
   T = T-0.1

draw_ising.setup(spin, tup, tdn)
################################################



# Run updates
while True :
   for i in range(updates_per_frame):
      x = int( L*np.random.random() )
      y = int( L*np.random.random() )

      # current energy
      energy_difference = 2*spin[x][y] * (
                            spin[x][(y+1)%L] + spin[x][y-1] 
                          + spin[(x+1)%L][y] + spin[x-1][y] )

      # propability from the distributions
      p = np.exp( -energy_difference/T )

      print("site ", x, y, p, energy_difference)

      rand = np.random.random()
      if rand < p:
         spin[x][y] = - spin[x][y]


   # Draw the current configuration
   draw_ising.draw(spin, T)
   
