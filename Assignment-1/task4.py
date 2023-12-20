"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the MultiBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, set_pulled, reward): This method is called 
        just after the give_pull method. The method should update the 
        algorithm's internal state based on the arm that was pulled and the 
        reward that was received.
        (The value of arm_index is the same as the one returned by give_pull 
        but set_pulled is the set that is randomly chosen when the pull is 
        requested from the bandit instance.)
"""

import numpy as np
import random
import math
# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE


class MultiBanditsAlgo:
    def __init__(self, num_arms, horizon):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.horizon = horizon
        # START EDITING HERE
        self.first_success = np.zeros(num_arms)
        self.second_success = np.zeros(num_arms)
        self.first_failure = np.zeros(num_arms)
        self.second_failure = np.zeros(num_arms)
        self.first_predict_values = np.zeros(num_arms)
        self.second_predict_values = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        for i in range(self.num_arms):
            self.first_predict_values[i] = random.betavariate(self.first_success[i]+1, self.first_failure[i]+1)
            self.second_predict_values[i] = random.betavariate(self.second_success[i]+1, self.second_failure[i]+1)
        return np.argmax(self.first_predict_values + self.second_predict_values)
        # END EDITING HERE
        return 0
    def get_reward(self, arm_index, set_pulled, reward):
        # START EDITING HERE
        if set_pulled == 0:
            if reward == 0:
                self.first_failure[arm_index] = self.first_failure[arm_index] + 1
            else :
                self.first_success[arm_index] = self.first_success[arm_index] + 1
        else:
            if reward == 0:
                self.second_failure[arm_index] = self.second_failure[arm_index] + 1
            else :
                self.second_success[arm_index] = self.second_success[arm_index] + 1 
        # END EDITING HERE
        return 0

