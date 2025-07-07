import numpy as np





class Agent:
   
   
    def __init__(self, n_arms=10, epsilon=0.1, step_size=None,c=0, g=0):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.step_size = step_size
        self.values = np.zeros(n_arms)
        self.count = np.zeros(n_arms)
        self.c = c
        self.H=np.zeros(n_arms)
        self.g=g
        self.pi= np.random.dirichlet(np.ones(n_arms))
        self.R=0

    def softmax(self):
        eH = np.exp(self.H - np.max(self.H))  # improve numerical stability
        return eH / eH.sum()
    
    def choose(self):
        if self.c==0 and self.g==0 :
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.n_arms)
            else:
                return np.argmax(self.values)
        elif self.g==0 :
            if self.count.sum()< self.n_arms :
                zero_indices = np.where(self.count == 0)[0]
                return np.random.choice(zero_indices)
            else :
                ucb_values = self.values + self.c * np.sqrt(np.log(self.count.sum()) / (self.count + 1e-5))
                return np.argmax(ucb_values)
        else:
            self.pi = self.softmax()
            if self.count.sum()< self.n_arms :
                zero_indices = np.where(self.count == 0)[0]
                return np.random.choice(zero_indices)
            else:
                return np.random.choice(self.n_arms, p=self.pi)

            
    
            
        
    

        

    def update(self, action, reward):
        if self.g==0:    
            self.count[action] += 1
            if self.step_size is None:
                alpha = 1 / self.count[action]
            else:
                alpha = self.step_size
            self.values[action] += alpha * (reward - self.values[action])
        else:
            self.count[action] += 1
            self.R += (reward-self.R)/self.count.sum()
            if self.step_size is None:
                alpha = 1 / self.count[action]
            else:
                alpha = self.step_size
            for x in range(self.n_arms):
                if x != action:
                    self.H[x] -= alpha* (reward - self.R)*self.pi[x]
                else:
                    self.H[action] += alpha* (reward - self.R)*self.pi[action]
            self.pi = self.softmax




        
    def reset(self):
        self.values = np.zeros(self.n_arms)
        self.count = np.zeros(self.n_arms)
        self.average_reward = 0.0
        self.reward_count = 0
        self.H = np.zeros(self.n_arms)
        self.pi = np.ones(self.n_arms) / self.n_arms
        self.R = 0.0
