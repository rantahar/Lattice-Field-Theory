#!/bin/python3

import numpy as np
import draw_ising 


### Set up parameters
lattice_size = 128
T = 1.0
updates_per_frame = 100

# Start with all spins 1
spin = np.ones([lattice_size,lattice_size], dtype=int) 



###########################################
# Assign the increase temperature buttons 
def tup(event):
   global T
   T = T+0.1

def tdn(event):
   global T
   T = T-0.1

draw_ising.setup(spin, tup, tdn)
###########################################



# Run updates
while True :
   for i in range(updates_per_frame):
      x = int( lattice_size*np.random.random() )
      y = int( lattice_size*np.random.random() )

      # change in energy
      energy_difference = 2*spin[x][y] * (
                            spin[x][(y+1)%lattice_size] + spin[x][y-1] 
                          + spin[(x+1)%lattice_size][y] + spin[x-1][y] )

      # propability from the distributions
      p = np.exp( -energy_difference/T )

      rand = np.random.random()
      if rand < p:
         spin[x][y] = - spin[x][y]


   ################################
   # Draw the current configuration
   draw_ising.draw(spin, T)
   ################################

   ## Add measurements here
  
   
