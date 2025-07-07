import numpy as np

class bandit:
    def __init__(self,n_arms=10):
        self.n_arms=n_arms
        self.true_q_values=np.random.normal(0,1,n_arms)
    def pull(self,arm):

        #reward is the reward from the arm + a random guassian noise mean 0 and std 1
        reward=self.true_q_values[arm]+np.random.normal(0,1)
        return reward
    