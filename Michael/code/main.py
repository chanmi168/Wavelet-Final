# %% Import modules

from dataParser import wavLoader
import numpy as np
import matplotlib.pyplot as plt
import os

# %% Loading data and labels

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, 'data/Speaker3/mike_0.wav')
rate, data = wavLoader(filename=my_file)

print(len(data))

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(data)



bbb = data[0:300]
print(rate)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(bbb)

plt.show()







# loop through all folders, import all files in a folder 




# %% Preporcess data: 
#   -> downsample   -> fragmentation    -> extract DWT coeff 
#   -> compute energy index

# %% LSTM implementation
