import matplotlib.pyplot as plt
import numpy as np
import glob

for file in sorted(glob.glob("Losses/loss_*.txt")):
    data = np.loadtxt(file)
    plt.plot(data[:,0], data[:,1], label=file.split('/')[-1].replace('.txt',''))

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss for Each Activation Function")
plt.legend()
plt.grid(True)
plt.show()
