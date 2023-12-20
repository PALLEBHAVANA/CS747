import sys
import numpy as np
from pulp import *


class mdp:
    def __init__(self):
        self.no_states = 0
        self.no_actions = 0
        self.num_terminal_states = 0
        self.terminal_states = []
        self.trans = []
        self.reward = []
        self.mdptype = ""
        self.discount = 0
        self.V_final = None
        self.p_final = None

    def vi(self):
        # TODO
        V_initial = np.zeros((self.no_states))
        self.V_final = np.zeros((self.no_states))
        while True:
            V_initial = self.V_final
            k = self.discount*V_initial
            k = self.trans * k
            self.V_final = np.amax(
                np.sum(self.reward, axis=2) + np.sum(k, axis=2), axis=1)
            error = np.linalg.norm(self.V_final-V_initial)
            if  error < 0.00000001:
                break
        k = self.discount*V_initial
        k = self.trans * k
        self.p_final = np.argmax(
            np.sum(self.reward, axis=2) + np.sum(k, axis=2), axis=1)
        try:
        	self.V_final = self.values(self.p_final)
        except:
        	self.V_final = self.V_final
        self.V_final = np.around(self.V_final, 6)
        for i in range(self.no_states):
            print(format(self.V_final[i], '.6f') + " " + str(self.p_final[i]))

    def improvable(self, policy):
        V_initial = self.values(policy)
        k = self.discount*V_initial
        k = self.trans * k
        return np.argmax(np.sum(self.reward, axis=2) + np.sum(k, axis=2), axis=1)

    def hpi(self):
        #TODO -- done
        self.p_final = np.random.randint(self.no_actions, size=self.no_states)
        while True:
            p_prev = self.p_final
            self.p_final = self.improvable(p_prev)
            if np.all(p_prev == self.p_final):
                break
        self.V_final = self.values(self.p_final)
        self.V_final = np.around(self.V_final, 6)
        for i in range(self.no_states):
            print(format(self.V_final[i], '.6f') + " " + str(self.p_final[i]))

    def lp(self):
        self.V_final = np.zeros((self.no_states))
        list_variables = []
        model = LpProblem("values", LpMinimize)
        list_variables = np.array(list_variables)
        N = range(self.no_states)
        list_variables = LpVariable.dicts("val", N, cat="continuous")
        model += lpSum(list_variables[i] for i in N)
        right_side1 = np.sum(self.reward, axis=2)
        for i in N:
            for j in range(self.no_actions):
                model += list_variables[i] >= (right_side1[i][j] + lpSum(self.discount * self.trans[i, j, k] * list_variables[k] for k in N))
        model.solve(PULP_CBC_CMD(msg=False))
        for i in range(self.no_states):
            self.V_final[i] = list_variables[i].varValue
        k = self.discount*self.V_final
        k = self.trans * k
        self.p_final = np.argmax(np.sum(self.reward, axis=2) + np.sum(k, axis=2), axis=1)
        self.V_final = self.values(self.p_final)
        self.V_final = np.around(self.V_final, 6)
        for i in range(self.no_states):
            print(format(self.V_final[i], '.6f') + " " + str(self.p_final[i]))

    def values(self, policy):
        equations = np.diag(np.full(self.no_states, 1))
        policy_indices = np.arange(len(policy))
        specific_transitions = self.trans[policy_indices, policy]
        specific_rewards = self.reward[policy_indices, policy]
        b = np.sum(specific_rewards, axis=1)
        k = self.discount * specific_transitions
        equations = equations - k
        values_policy = np.linalg.solve(equations, b)
        return values_policy


n = len(sys.argv)

if n < 3:
    print("Invalid arguments")
    sys.exit(1)
if sys.argv[1] != "--mdp":
    print("second argument should be --mdp")
    sys.exit(1)
file_path = sys.argv[2]
mdp_instance = mdp()
file1 = open(file_path, 'r')
Lines = file1.readlines()
for line in Lines:
    li = line.split()
    if li[0] == "numStates":
        mdp_instance.no_states = int(li[1])
    elif li[0] == "numActions":
        mdp_instance.no_actions = int(li[1])
        mdp_instance.trans = np.zeros(
            (mdp_instance.no_states, mdp_instance.no_actions, mdp_instance.no_states))
        mdp_instance.reward = np.zeros(
            (mdp_instance.no_states, mdp_instance.no_actions, mdp_instance.no_states))
    elif li[0] == "end" and li[1] != "-1":
        li2 = li[1:]
        terminal_states = list(np.array(li2, dtype=int))
        mdp_instance.num_terminal_states = len(terminal_states)
    elif li[0] == "transition":
        li2 = li[1:]
        transition = list(np.array(li2, dtype=float))
        mdp_instance.trans[int(transition[0])][int(
            transition[1])][int(transition[2])] = transition[4]
        mdp_instance.reward[int(transition[0])][int(
            transition[1])][int(transition[2])] = transition[3]*transition[4]
    elif li[0] == "mdptype":
        mdp_instance.mdptype = li[1]
    elif li[0] == "discount":
        mdp_instance.discount = float(li[1])

default = "vi"
if n > 4:
    if sys.argv[3] == "--algorithm":
        default = sys.argv[4]
        if default == "vi":
            # TODO make a vi function call here. -- done
            mdp_instance.vi()
        if default == "hpi":
            # TODO make a hpi function call here.
            mdp_instance.hpi()
        if default == "lp":
            # TODO make a lp function call here.
            mdp_instance.lp()
    elif sys.argv[3] == "--policy":
        # TODO make a function call here. - done
        policy_file = sys.argv[4]
        file1 = open(policy_file, 'r')
        Lines = file1.readlines()
        policy = np.array(Lines, dtype=int)
        values = mdp_instance.values(policy)
        values = np.around(values, 6)
        for i in range(mdp_instance.no_states):
            print(format(values[i], '.6f') + " " + str(policy[i]))
else:
    mdp_instance.vi()
