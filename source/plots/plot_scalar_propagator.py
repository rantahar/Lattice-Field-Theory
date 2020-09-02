import matplotlib.pyplot as plt
import numpy as np

k = np.linspace(-3.14, 3.14, num=200)
m=0.1

fig, ax = plt.subplots()

sink = np.sin(k/2)
k0_lattice = 2*np.arcsinh(np.sqrt(sink**2+m**2/4))
k0_continuum = np.sqrt(k**2+m**2)
plt.plot(k, k0_lattice, label='lattice')
plt.plot(k, k0_continuum, label='lattice')

plt.xlabel(r'$|\mathbf{k}|$')
plt.ylabel(r'$k_0$')
ax.legend()
plt.show()

