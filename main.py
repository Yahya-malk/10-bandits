import numpy as np
from agent import Agent
from bandit import bandit 
import matplotlib.pyplot as plt 
def main():
    #AGENT1 goes for greedy approach
    n_bandit = 2000
    n_rounds = 1000
    all_rewards = np.zeros((n_bandit, n_rounds))    
    all_optimal = np.zeros((n_bandit, n_rounds))
    agent1= Agent(n_arms=10, epsilon=0)
    for i in range(n_bandit):
        # Create a bandit instance
        b = bandit(n_arms=10)
        for j in range(n_rounds):
            action=agent1.choose()
            reward=b.pull(action)
            agent1.update(action,reward)
            all_rewards[i,j] = reward
            all_optimal[i,j] = 1 if action == np.argmax(b.true_q_values) else 0
        agent1.reset()
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_optimal = np.mean(all_optimal, axis=0)
    np.save("avg_rewards.npy", avg_rewards)
    np.save("avg_optimal.npy", avg_optimal)
    
    #agent2&3 goes for epsilon greedy approach
    n_bandit = 2000
    n_rounds = 1000
    all_rewards = np.zeros((n_bandit, n_rounds))    
    all_optimal = np.zeros((n_bandit, n_rounds))
    agent2= Agent(n_arms=10, epsilon=0.01)
    for i in range(n_bandit):
        # Create a bandit instance
        b = bandit(n_arms=10)
        for j in range(n_rounds):
            action=agent2.choose()
            reward=b.pull(action)
            agent2.update(action,reward)
            all_rewards[i,j] = reward
            all_optimal[i,j] = 1 if action == np.argmax(b.true_q_values) else 0
        agent2.reset()
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_optimal = np.mean(all_optimal, axis=0)
    np.save("avg_rewards2.npy", avg_rewards)
    np.save("avg_optimal2.npy", avg_optimal)

  
    n_bandit = 2000
    n_rounds = 1000
    all_rewards = np.zeros((n_bandit, n_rounds))    
    all_optimal = np.zeros((n_bandit, n_rounds))
    agent3= Agent(n_arms=10, epsilon=0.1)
    for i in range(n_bandit):
        # Create a bandit instance
        b = bandit(n_arms=10)
        for j in range(n_rounds):
            action=agent3.choose()
            reward=b.pull(action)
            agent3.update(action,reward)
            all_rewards[i,j] = reward
            all_optimal[i,j] = 1 if action == np.argmax(b.true_q_values) else 0
        agent3.reset()
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_optimal = np.mean(all_optimal, axis=0)
    np.save("avg_rewards3.npy", avg_rewards)
    np.save("avg_optimal3.npy", avg_optimal)

    n_bandit = 2000
    n_rounds = 1000
    all_rewards = np.zeros((n_bandit, n_rounds))    
    all_optimal = np.zeros((n_bandit, n_rounds))
    ucb_agent= Agent(n_arms=10, epsilon=0, c=2)
    for i in range(n_bandit):
        # Create a bandit instance
        b = bandit(n_arms=10)
        for j in range(n_rounds):
            action=ucb_agent.choose()
            reward=b.pull(action)
            ucb_agent.update(action,reward)
            all_rewards[i,j] = reward
            all_optimal[i,j] = 1 if action == np.argmax(b.true_q_values) else 0
        ucb_agent.reset()
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_optimal = np.mean(all_optimal, axis=0)
    np.save("avg_rewards_ucb.npy", avg_rewards)
    np.save("avg_optimal_ucb.npy", avg_optimal)


    n_bandit = 2000
    n_rounds = 1000
    all_rewards = np.zeros((n_bandit, n_rounds))    
    all_optimal = np.zeros((n_bandit, n_rounds))
    gradient_agent= Agent(n_arms=10, epsilon=0, g=1)
    for i in range(n_bandit):
        # Create a bandit instance
        b = bandit(n_arms=10)


        if i % 500 == 0 and j == 0:
            print(f"π at bandit {i}: {gradient_agent.pi}")
        for j in range(n_rounds):
            action=gradient_agent.choose()
            reward=b.pull(action)
            gradient_agent.update(action,reward)
            all_rewards[i,j] = reward
            all_optimal[i,j] = 1 if action == np.argmax(b.true_q_values) else 0
        gradient_agent.reset()
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_optimal = np.mean(all_optimal, axis=0)
    np.save("avg_rewards_gradient.npy", avg_rewards)
    np.save("avg_optimal_gradient.npy", avg_optimal)
    

if __name__ == "__main__":
    main()