import matplotlib.pyplot as plt
import numpy as np
import glob

for file in sorted(glob.glob("Losses/loss_*.txt")):
    data = np.loadtxt(file)
    act_name = file.split('/')[-1].replace('.txt', '').replace('loss_', '')

    plt.plot(data[:,0], data[:,1], label=f"train_{act_name}")
    plt.plot(data[:,0], data[:,2], '--', label=f"eval_{act_name}")

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training & Evaluation Loss for Each Activation Function")
plt.legend()
plt.grid(True)
plt.show()
