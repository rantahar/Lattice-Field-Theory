import matplotlib.pyplot as plt
import numpy as np

h=0.05
phase1 = np.linspace(0.0, 1.0, num=50)
phase2 = np.linspace(1.0, 2.0, num=50)

fig, ax = plt.subplots()

E1 = 2-0.3*phase1 + 0.1*phase1**2
E2 = 1.2-0.1*phase2 - 0.1*phase1**2
plt.plot(phase1, E1, label='E', color='C0')
plt.plot([phase1[-1],phase2[0]], [E1[-1],E2[0]], color='C0', linestyle='--')
plt.plot(phase2, E2, color='C0')


plt.xlabel(r'$T$')
ax.legend()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.show()

