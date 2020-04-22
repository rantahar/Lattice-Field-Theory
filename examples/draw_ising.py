#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button



#################################
# For displaying the simulation #

# Create the image window
fig, ax = plt.subplots()
palette = np.array([[255,   0,   0],   # red
                    [  0, 255,   0]])  # green

def stop(event):
   print("stop")
   quit()


def setup(spin,tup,tdn):
   global im, temperature_text, bstop, button_tup, button_tdn
   image = spin.copy()
   image[image<0] = 0
   image = palette[image]
   im = plt.imshow(image)

   temperature_text = plt.text(1.1, 0.50, "", transform=ax.transAxes)

   axstop = plt.axes([0.85, 0.05, 0.1, 0.075])
   bstop = Button(axstop, 'Stop')
   bstop.on_clicked(stop)

   tup_button_location = plt.axes([0.85, 0.4, 0.1, 0.075])
   button_tup = Button(tup_button_location, '+')
   button_tup.on_clicked(tup)

   tdn_button_location = plt.axes([0.85, 0.33, 0.1, 0.075])
   button_tdn = Button(tdn_button_location, '-')
   button_tdn.on_clicked(tdn)



def draw(spin, T):
   image = spin.copy()
   image[image<0] = 0
   image = palette[image]

   temperature_text.set_text("T = " + str(T))

   im.set_data(image)
   plt.draw()
   plt.pause(0.01)





