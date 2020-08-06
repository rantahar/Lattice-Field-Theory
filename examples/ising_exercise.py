import numpy as np

lattice_size = 16
temperature = 4.0
number_of_updates = 1000

spin = np.ones([lattice_size,lattice_size], dtype=int)

for i in range(number_of_updates):
  x = int( lattice_size*np.random.random() )
  y = int( lattice_size*np.random.random() )

  energy_difference = 2*spin[x][y] * (
                    spin[x][(y+1)%lattice_size] + spin[x][y-1]
                  + spin[(x+1)%lattice_size][y] + spin[x-1][y] )
  probability = np.exp( -energy_difference/temperature )

  if np.random.random() < probability:
   spin[x][y] = - spin[x][y]

  energy = 0
  magnetisation = 0
  for x in range(lattice_size):
    for y in range(lattice_size):
        energy += 4 - spin[x][y] * (
                      spin[x][(y+1)%lattice_size] + spin[x][y-1]
                    + spin[(x+1)%lattice_size][y] + spin[x-1][y] )
        magnetisation += spin[x][y]

  print("The energy is {}".format(energy))
  print("The magnetisation is {}".format(magnetisation / (lattice_size*lattice_size)))

