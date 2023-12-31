"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the FaultyBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon, fault): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

import numpy as np
import random
# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class FaultyBanditsAlgo:
    def __init__(self, num_arms, horizon, fault):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.horizon = horizon
        self.fault = fault # probability that the bandit returns a faulty pull
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
        pa = (self.success_counts[arm_index] + 1)/ (self.success_counts[arm_index] + self.failure_counts[arm_index]+2)
        success = pa *((1-self.fault)*reward + 0.5*self.fault)/ (pa *((1-self.fault)*reward + 0.5*self.fault) + (1-pa)*((1-self.fault)*(1-reward) + 0.5*self.fault))
        failure = 1-success
        self.failure_counts[arm_index] = self.failure_counts[arm_index] +  failure
        self.success_counts[arm_index] = self.success_counts[arm_index]  + success
        #END EDITING HERE

