"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
import random
from bisect import bisect_left  
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon

    def give_pull(self):
        raise NotImplementedError

    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)

    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)

    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

# START EDITING HERE
# You can use this space to define any helper functions that you need
def ucb_value(p,  k):
    l = p
    r = 1
    while r - l > 0.01:
    	x = (r + l)/2
    	value = - p*math.log(x) - (1-p)*math.log(1-x)
    	if(  value <= k ):
        	l = x
    	else:
        	r = x
    return (l+r)/2


# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.ucb_values = np.zeros(num_arms) #stores the UCB values. 
        self.probability_values = np.zeros(num_arms) #stores the probability values.
        self.counts = np.zeros(num_arms) # total number of times each arm is pulled.
        self.time = 0 # time instance.
        self.num_arms = num_arms #total number of arms.
        # END EDITING HERE

    def give_pull(self):
        # START EDITING HERE
        k = self.ucb_values # get the UCB values
        self.time = self.time + 1
        if self.time <= self.num_arms: # If there is not explored one, prefer it as the more priority, 
            return self.time - 1# return first element from that
        else:
            numerator = math.sqrt(2*math.log(self.time))
            for i in range(self.num_arms): # update this for every arm.
                self.ucb_values[i] = self.probability_values[i] + (numerator/math.sqrt(self.counts[i])) #calculate the ucb value.
            return np.argmax(self.ucb_values) #return maximum ucb_value one.
        # END EDITING HERE  


    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1 # increase the count of the arm.
        n = self.counts[arm_index] # get number of times it is pulled until now.
        self.probability_values[arm_index] = ((n - 1) /float(n)) * self.probability_values[arm_index] + (1 / float(n)) * reward # update the probability of the arm.
        #print(self.ucb_values)
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.kl_ucb_values = np.zeros(num_arms) # KL_UCB values.
        self.probability_values = np.zeros(num_arms) # probability values.
        self.counts = np.zeros(num_arms) # Counts of each arm.
        self.time = 0  # time instance.
        self.num_arms = num_arms # number of arms.
        # END EDITING HERE

    def give_pull(self):
        # START EDITING HERE
        self.time = self.time + 1 # Increase the time.
        if self.time  <= self.num_arms: # pull each arm atleast once.
            return self.time - 1 
        else:
            for i in range(self.num_arms):
                numerator = (math.log(self.time))
                p = self.probability_values[i]
                k = numerator/self.counts[i]
                if p != 0 and p != 1: # because the value becomes undefined.
                    k -= (p*math.log(p)+(1-p)*math.log(1-p))
                self.kl_ucb_values[i] = ucb_value(p, k)
            return np.argmax(self.kl_ucb_values) # else return according to the rule.
        # END EDITING HERE

    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] = self.counts[arm_index] + 1
        n = self.counts[arm_index]
        self.probability_values[arm_index] = ((n - 1) / float(n)) * self.probability_values[arm_index] + (1 / float(n)) * reward
        #print(self.kl_ucb_values)
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.success_counts = np.zeros(num_arms)
        self.failure_counts = np.zeros(num_arms)
        self.predict_values = np.zeros(num_arms)
        # END EDITING HERE

    def give_pull(self):
        # START EDITING HERE
        for i in range(self.num_arms):
            self.predict_values[i] = random.betavariate(self.success_counts[i]+1, self.failure_counts[i]+1)
        return np.argmax(self.predict_values)
        # END EDITING HERE

    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward == 0:
            self.failure_counts[arm_index]  = self.failure_counts[arm_index] + 1
        else:
            self.success_counts[arm_index] = self.success_counts[arm_index] + 1
        # END EDITING HERE

