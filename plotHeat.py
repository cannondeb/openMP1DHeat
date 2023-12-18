import matplotlib.pyplot as plt
import numpy as np

T = np.loadtxt("T.txt")

plt.imshow(T, cmap='rainbow')
plt.colorbar()
plt.show()