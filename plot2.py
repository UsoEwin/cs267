import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot([2,3,4,5], [3.3,4.2,5.7,8.2], 'r')
plt.plot([2,3,4,5], [4.6,8.2,11.2,14], 'k')
plt.axis([1, 6, 2, 15])
fig.suptitle('Speedup for different searching levels', fontsize=20)
plt.xlabel('Levels', fontsize=18)
plt.ylabel('Speedup', fontsize=16)

plt.legend(['Johnson2016', 'My result'], loc='upper left')

plt.show()