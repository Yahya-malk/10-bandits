import numpy as np
import matplotlib.pyplot as plt

# Load ε-greedy results
avg_rewards1 = np.load("avg_rewards.npy")
avg_optimal1 = np.load("avg_optimal.npy")

avg_rewards2 = np.load("avg_rewards2.npy")
avg_optimal2 = np.load("avg_optimal2.npy")

avg_rewards3 = np.load("avg_rewards3.npy")
avg_optimal3 = np.load("avg_optimal3.npy")

# Load UCB results
avg_rewards_ucb = np.load("avg_rewards_ucb.npy")
avg_optimal_ucb = np.load("avg_optimal_ucb.npy")

# Load Gradient Bandit results
avg_rewards_grad = np.load("avg_rewards_gradient.npy")
avg_optimal_grad = np.load("avg_optimal_gradient.npy")

# Create figure
plt.figure(figsize=(14,6))

# Subplot 1: Average Reward
plt.subplot(1, 2, 1)
plt.plot(avg_rewards1, label="ε=0 (Greedy)")
plt.plot(avg_rewards2, label="ε=0.01")
plt.plot(avg_rewards3, label="ε=0.1")
plt.plot(avg_rewards_ucb, label="UCB (c=2)", linestyle="--")
plt.plot(avg_rewards_grad, label="Gradient Bandit", linestyle=":")
plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.title("Average Reward per Step")
plt.legend()

# Subplot 2: Optimal Action %
plt.subplot(1, 2, 2)
plt.plot(avg_optimal1, label="ε=0 (Greedy)")
plt.plot(avg_optimal2, label="ε=0.01")
plt.plot(avg_optimal3, label="ε=0.1")
plt.plot(avg_optimal_ucb, label="UCB (c=2)", linestyle="--")
plt.plot(avg_optimal_grad, label="Gradient Bandit", linestyle=":")
plt.xlabel("Step")
plt.ylabel("Optimal Action %")
plt.title("Probability of Selecting Optimal Action")
plt.legend()

plt.tight_layout()
plt.savefig("bandit_results_all_strategies.png", dpi=300, bbox_inches="tight")
plt.show()
