import matplotlib.pyplot as plt
import numpy as np

h=0.05
b = np.linspace(1e-5, 2.0, num=50)

fig, ax = plt.subplots()

for h in [0.01,0.1,0.5]:
    M = np.sinh(h)/(np.sqrt(np.sinh(h)**2 + np.exp(-4*(1/b))))
    plt.plot(b, M, label='h='+str(h))

plt.xlabel(r'$T=1/\beta$')
plt.ylabel('M')
ax.legend()
plt.show()

