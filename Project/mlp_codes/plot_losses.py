import matplotlib.pyplot as plt
import numpy as np
import glob

# 🔹 Ask user for learning rate type
choice = int(input("Choisissez le type de learning rate (0=fixed, 1=inverse, 2=exponential) : "))

if choice == 0:
    lr_name = "Fixed Learning Rate"
elif choice == 1:
    lr_name = "Inverse Time Decay"
elif choice == 2:
    lr_name = "Exponential Decay"
else:
    lr_name = "Unknown Schedule"

# 🔹 Create subplots: 1 row, 2 columns
fig, (ax_train, ax_eval) = plt.subplots(1, 2, figsize=(12, 5))

# 🔹 Read loss files
for file in sorted(glob.glob("Losses/loss_*.txt")):
    data = np.loadtxt(file)
    act_name = file.split('/')[-1].replace('.txt', '').replace('loss_', '')

    # Plot training loss
    ax_train.plot(data[:, 0], data[:, 1], label=f"train_{act_name}")

    # Plot evaluation loss
    ax_eval.plot(data[:, 0], data[:, 2], '--', label=f"eval_{act_name}")

# 🔹 Customize training plot
ax_train.set_xlabel("Iteration")
ax_train.set_ylabel("Loss")
ax_train.set_title("Training Loss")
ax_train.set_yscale("log", base=10)
ax_train.grid(True, which="both", ls="--")
ax_train.legend()

# 🔹 Customize evaluation plot
ax_eval.set_xlabel("Iteration")
ax_eval.set_ylabel("Loss")
ax_eval.set_title("Evaluation Loss")
ax_eval.set_yscale("log", base=10)
ax_eval.grid(True, which="both", ls="--")
ax_eval.legend()

# 🔹 Add global title
fig.suptitle(f"Training & Evaluation Loss ({lr_name})", fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
