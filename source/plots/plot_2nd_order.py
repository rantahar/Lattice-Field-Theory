import matplotlib.pyplot as plt
import numpy as np

h=0.05
phase1 = np.linspace(0.0, 1.0, num=50)
phase2 = np.linspace(1.0, 2.0, num=50)

fig, ax = plt.subplots()

M1 = 0*phase1
M2 = 0.2*(phase2-1.0)**0.5
plt.plot(phase1, M1, color='C0')
plt.plot(phase2, M2, color='C0')

plt.xlabel(r'$T$')
plt.ylabel('M')
plt.show()

