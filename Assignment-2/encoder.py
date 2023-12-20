# Encode.
import numpy as np
import sys

#Encoding.
def encode(b1, b2, r, poss):
    return (b1*16**2 + b2 * 16 + r)*2 + poss

#Transitions.
encoded_transitions = np.zeros((16*16*16*2+1, 10, 16*16*16*2+1))
number_encoded = np.zeros((16, 16, 16, 2))
for b1 in range(16):
    for b2 in range(16):
        for r1 in range(16):
            for poss in range(2):
                number_encoded[b1, b2, r1, poss] = encode(b1, b2, r1, poss)
                
#Taking the policy of apponent into trans_r
trans_r = np.zeros((16, 16, 16, 2, 4))
policy_path = sys.argv[2]
file1 = open(policy_path, 'r')
Lines = file1.readlines()[1:]
for line in Lines:
    li = line.split()
    k = li[0]
    i = 0
    i1 = int(k[0:2])-1
    i2 = int(k[2:4])-1
    i3 = int(k[4:6])-1
    i4 = int(k[-1])-1
    for p in li[1:]:
        trans_r[i1][i2][i3][i4][i] = float(p)
        i = i+1

#p and q for the players
p = float(sys.argv[4])
q = float(sys.argv[6])

# Transition probabilities,
trans = np.zeros((16, 16, 16, 2, 10))
reward = np.zeros((16, 16, 16, 2, 10))

# Movement.
trans[:, :, :, 0, :4] = 1-2*p
trans[:, :, :, 1, :4] = 1-p
trans[:, :, :, 0, 4:8] = 1-p
trans[:, :, :, 1, 4:8] = 1-2*p
# Fixing the not probable movements.
# R1
trans[0:4, :, :, :, 2] = 0
trans[3:16:4, :, :, :, 1] = 0
trans[12:16, :, :, :, 3] = 0
trans[0:13:4, :, :, :, 0] = 0
# R2
trans[:, 0:4, :, :, 6] = 0
trans[:, 3:16:4, :, :, 5] = 0
trans[:, 12:16, :, :, 7] = 0
trans[:, 0:13:4, :, :, 4] = 0

prob = np.zeros((16, 16, 16, 2, 10))
# 8193 states. 16*16*16*2 states are for normal transitions, last one is for acceptance. (8192 for acceptance)
transitions = np.zeros((16, 16, 16, 2, 10, 16, 16, 16, 2))
for b1 in range(16):
    for b2 in range(16):
        for r in range(16):
            for poss in range(2):
                x1 = int(number_encoded[b1, b2, r, poss])
                if b1 > 0:
                    if r > 0:
                        transitions[b1, b2, r, poss, 0, b1-1, b2, r-1, poss] = trans[b1, b2, r, poss, 0] * trans_r[b1, b2, r, poss, 0]
                        x2 = int(number_encoded[b1-1, b2, r-1, poss])
                        encoded_transitions[x1, 0, x2] = transitions[b1, b2, r, poss, 0, b1-1, b2, r-1, poss]
                    if r < 15:
                        transitions[b1, b2, r, poss, 0, b1-1, b2, r+1, poss] = trans[b1, b2, r, poss, 0] * trans_r[b1, b2, r, poss, 1]
                        x2 = int(number_encoded[b1-1, b2, r+1, poss])
                        encoded_transitions[x1, 0, x2] = transitions[b1, b2, r, poss, 0, b1-1, b2, r+1, poss]
                    if r > 3:
                        transitions[b1, b2, r, poss, 0, b1-1, b2, r-4, poss] = trans[b1, b2, r, poss, 0] * trans_r[b1, b2, r, poss, 2]
                        x2 = int(number_encoded[b1-1, b2, r-4, poss])
                        encoded_transitions[x1, 0, x2] = transitions[b1, b2, r, poss, 0, b1-1, b2, r-4, poss]
                    if r < 12:
                        transitions[b1, b2, r, poss, 0, b1-1, b2, r+4, poss] = trans[b1, b2, r, poss, 0] * trans_r[b1, b2, r, poss, 3]
                        x2 = int(number_encoded[b1-1, b2, r+4, poss])
                        encoded_transitions[x1, 0, x2] = transitions[b1, b2, r, poss, 0, b1-1, b2, r+4, poss]
                if b1 < 15:
                    if r > 0:
                        transitions[b1, b2, r, poss, 1, b1+1, b2, r-1, poss] = trans[b1, b2, r, poss, 1] * trans_r[b1, b2, r, poss, 0]
                        x2 = int(number_encoded[b1+1, b2, r-1, poss])
                        encoded_transitions[x1, 1, x2] = transitions[b1, b2, r, poss, 1, b1+1, b2, r-1, poss]
                    if r < 15:
                        transitions[b1, b2, r, poss, 1, b1+1, b2, r+1, poss] = trans[b1, b2, r, poss, 1] * trans_r[b1, b2, r, poss, 1]
                        x2 = int(number_encoded[b1+1, b2, r+1, poss])
                        encoded_transitions[x1, 1, x2] = transitions[b1, b2, r, poss, 1, b1+1, b2, r+1, poss]
                    if r > 3:
                        transitions[b1, b2, r, poss, 1, b1+1, b2, r-4, poss] = trans[b1, b2, r, poss, 1] * trans_r[b1, b2, r, poss, 2]
                        x2 = int(number_encoded[b1+1, b2, r-4, poss])
                        encoded_transitions[x1, 1, x2] = transitions[b1, b2, r, poss, 1, b1+1, b2, r-4, poss]
                    if r < 12:
                        transitions[b1, b2, r, poss, 1, b1+1, b2, r+4, poss] = trans[b1, b2, r, poss, 1] * trans_r[b1, b2, r, poss, 3]
                        x2 = int(number_encoded[b1+1, b2, r+4, poss])
                        encoded_transitions[x1, 1, x2] = transitions[b1, b2, r, poss, 1, b1+1, b2, r+4, poss]

                if b1 > 3:
                    if r > 0:
                        transitions[b1, b2, r, poss, 2, b1-4, b2, r-1, poss] = trans[b1, b2, r, poss, 2] * trans_r[b1, b2, r, poss, 0]
                        x2 = int(number_encoded[b1-4, b2, r-1, poss])
                        encoded_transitions[x1, 2, x2] = transitions[b1, b2, r, poss, 2, b1-4, b2, r-1, poss]
                    if r < 15:
                        transitions[b1, b2, r, poss, 2, b1-4, b2, r+1, poss] = trans[b1, b2, r, poss, 2] * trans_r[b1, b2, r, poss, 1]
                        x2 = int(number_encoded[b1-4, b2, r+1, poss])
                        encoded_transitions[x1, 2, x2] = transitions[b1, b2, r, poss, 2, b1-4, b2, r+1, poss]
                    if r > 3:
                        transitions[b1, b2, r, poss, 2, b1-4, b2, r-4, poss] = trans[b1, b2, r, poss, 2] * trans_r[b1, b2, r, poss, 2]
                        x2 = int(number_encoded[b1-4, b2, r-4, poss])
                        encoded_transitions[x1, 2, x2] = transitions[b1, b2, r, poss, 2, b1-4, b2, r-4, poss]
                    if r < 12:
                        transitions[b1, b2, r, poss, 2, b1-4, b2, r+4, poss] = trans[b1, b2, r, poss, 2] * trans_r[b1, b2, r, poss, 3]
                        x2 = int(number_encoded[b1-4, b2, r+4, poss])
                        encoded_transitions[x1, 2, x2] = transitions[b1, b2, r, poss, 2, b1-4, b2, r+4, poss]
                if b1 < 12:
                    if r > 0:
                        transitions[b1, b2, r, poss, 3, b1+4, b2, r-1, poss] = trans[b1, b2, r, poss, 3] * trans_r[b1, b2, r, poss, 0]
                        x2 = int(number_encoded[b1+4, b2, r-1, poss])
                        encoded_transitions[x1, 3, x2] = transitions[b1, b2, r, poss, 3, b1+4, b2, r-1, poss]
                    if r < 15:
                        transitions[b1, b2, r, poss, 3, b1+4, b2, r+1, poss] = trans[b1, b2, r, poss, 3] * trans_r[b1, b2, r, poss, 1]
                        x2 = int(number_encoded[b1+4, b2, r+1, poss])
                        encoded_transitions[x1, 3, x2] = transitions[b1, b2, r, poss, 3, b1+4, b2, r+1, poss]
                    if r > 3:
                        transitions[b1, b2, r, poss, 3, b1+4, b2, r-4, poss] = trans[b1, b2, r, poss, 3] * trans_r[b1, b2, r, poss, 2]
                        x2 = int(number_encoded[b1+4, b2, r-4, poss])
                        encoded_transitions[x1, 3, x2] = transitions[b1, b2, r, poss, 3, b1+4, b2, r-4, poss]
                    if r < 12:
                        transitions[b1, b2, r, poss, 3, b1+4, b2, r+4, poss] = trans[b1, b2, r, poss, 3] * trans_r[b1, b2, r, poss, 3]
                        x2 = int(number_encoded[b1+4, b2, r+4, poss])
                        encoded_transitions[x1, 3, x2] = transitions[b1,b2, r, poss, 3, b1+4, b2, r+4, poss]
                if b2 > 0:
                    if r > 0:
                        transitions[b1, b2, r, poss, 4, b1, b2-1, r-1, poss] = trans[b1, b2, r, poss, 4] * trans_r[b1, b2, r, poss, 0]
                        x2 = int(number_encoded[b1, b2-1, r-1, poss])
                        encoded_transitions[x1, 4, x2] = transitions[b1, b2, r, poss, 4, b1, b2-1, r-1, poss]
                    if r < 15:
                        transitions[b1, b2, r, poss, 4, b1, b2-1, r+1, poss] = trans[b1, b2, r, poss, 4] * trans_r[b1, b2, r, poss, 1]
                        x2 = int(number_encoded[b1, b2-1, r+1, poss])
                        encoded_transitions[x1, 4, x2] = transitions[b1, b2, r, poss, 4, b1, b2-1, r+1, poss]
                    if r > 3:
                        transitions[b1, b2, r, poss, 4, b1, b2-1, r-4, poss] = trans[b1, b2, r, poss, 4] * trans_r[b1, b2, r, poss, 2]
                        x2 = int(number_encoded[b1, b2-1, r-4, poss])
                        encoded_transitions[x1, 4, x2] = transitions[b1, b2, r, poss, 4, b1, b2-1, r-4, poss]
                    if r < 12:
                        transitions[b1, b2, r, poss, 4, b1, b2-1, r+4, poss] = trans[b1, b2, r, poss, 4] * trans_r[b1, b2, r, poss, 3]
                        x2 = int(number_encoded[b1, b2-1, r+4, poss])
                        encoded_transitions[x1, 4, x2] = transitions[b1, b2, r, poss, 4, b1, b2-1, r+4, poss]

                if b2 < 15:
                    if r > 0:
                        transitions[b1, b2, r, poss, 5, b1, b2+1, r-1, poss] = trans[b1, b2, r, poss, 5] * trans_r[b1, b2, r, poss, 0]
                        x2 = int(number_encoded[b1, b2+1, r-1, poss])
                        encoded_transitions[x1, 5, x2] = transitions[b1, b2, r, poss, 5, b1, b2+1, r-1, poss]
                    if r < 15:
                        transitions[b1, b2, r, poss, 5, b1, b2+1, r+1, poss] = trans[b1, b2, r, poss, 5] * trans_r[b1, b2, r, poss, 1]
                        x2 = int(number_encoded[b1, b2+1, r+1, poss])
                        encoded_transitions[x1, 5, x2] = transitions[b1, b2, r, poss, 5, b1, b2+1, r+1, poss]
                    if r > 3:
                        transitions[b1, b2, r, poss, 5, b1, b2+1, r-4, poss] = trans[b1, b2, r, poss, 5] * trans_r[b1, b2, r, poss, 2]
                        x2 = int(number_encoded[b1, b2+1, r-4, poss])
                        encoded_transitions[x1, 5, x2] = transitions[b1, b2, r, poss, 5, b1, b2+1, r-4, poss]
                    if r < 12:
                        transitions[b1, b2, r, poss, 5, b1, b2+1, r+4, poss] = trans[b1, b2, r, poss, 5] * trans_r[b1, b2, r, poss, 3]
                        x2 = int(number_encoded[b1, b2+1, r+4, poss])
                        encoded_transitions[x1, 5, x2] = transitions[b1, b2, r, poss, 5, b1, b2+1, r+4, poss]

                if b2 > 3:
                    if r > 0:
                        transitions[b1, b2, r, poss, 6, b1, b2-4, r-1, poss] = trans[b1, b2, r, poss, 6] * trans_r[b1, b2, r, poss, 0]
                        x2 = int(number_encoded[b1, b2-4, r-1, poss])
                        encoded_transitions[x1, 6, x2] = transitions[b1, b2, r, poss, 6, b1, b2-4, r-1, poss]
                    if r < 15:
                        transitions[b1, b2, r, poss, 6, b1, b2-4, r+1, poss] = trans[b1, b2, r, poss, 6] * trans_r[b1, b2, r, poss, 1]
                        x2 = int(number_encoded[b1, b2-4, r+1, poss])
                        encoded_transitions[x1, 6, x2] = transitions[b1, b2, r, poss, 6, b1, b2-4, r+1, poss]
                    if r > 3:
                        transitions[b1, b2, r, poss, 6, b1, b2-4, r-4, poss] = trans[b1, b2, r, poss, 6] * trans_r[b1, b2, r, poss, 2]
                        x2 = int(number_encoded[b1, b2-4, r-4, poss])
                        encoded_transitions[x1, 6, x2] = transitions[b1, b2, r, poss, 6, b1, b2-4, r-4, poss]
                    if r < 12:
                        transitions[b1, b2, r, poss, 6, b1, b2-4, r+4, poss] = trans[b1, b2, r, poss, 6] * trans_r[b1, b2, r, poss, 3]
                        x2 = int(number_encoded[b1, b2-4, r+4, poss])
                        encoded_transitions[x1, 6, x2] = transitions[b1, b2, r, poss, 6, b1, b2-4, r+4, poss]
                if b2 < 12:
                    if r > 0:
                        transitions[b1, b2, r, poss, 7, b1, b2+4, r-1, poss] = trans[b1, b2, r, poss, 7] * trans_r[b1, b2, r, poss, 0]
                        x2 = int(number_encoded[b1, b2+4, r-1, poss])
                        encoded_transitions[x1, 7, x2] = transitions[b1, b2, r, poss, 7, b1, b2+4, r-1, poss]
                    if r < 15:
                        transitions[b1, b2, r, poss, 7, b1, b2+4, r+1, poss] = trans[b1, b2, r, poss, 7] * trans_r[b1, b2, r, poss, 1]
                        x2 = int(number_encoded[b1, b2+4, r+1, poss])
                        encoded_transitions[x1, 7, x2] = transitions[b1, b2, r, poss, 7, b1, b2+4, r+1, poss]
                    if r > 3:
                        transitions[b1, b2, r, poss, 7, b1, b2+4, r-4, poss] = trans[b1, b2, r, poss, 7] * trans_r[b1, b2, r, poss, 2]
                        x2 = int(number_encoded[b1, b2+4, r-4, poss])
                        encoded_transitions[x1, 7, x2] = transitions[b1, b2, r, poss, 7, b1, b2+4, r-4, poss]
                    if r < 12:
                        transitions[b1, b2, r, poss, 7, b1, b2+4, r+4, poss] = trans[b1, b2, r, poss, 7] * trans_r[b1, b2, r, poss, 3]
                        x2 = int(number_encoded[b1, b2+4, r+4, poss])
                        encoded_transitions[x1, 7, x2] = transitions[b1, b2, r, poss, 7, b1, b2+4, r+4, poss]


# Tackling.

moved = []
for i in range(16):
    l1 = []

    if i < 15:
        l1.append(i+1)
    else:
        l1.append(-1)
    if i > 0:
        l1.append(i-1)
    else:
        l1.append(-1)
    if i < 12:
        l1.append(i+4)
    else:
        l1.append(-1)
    if i > 3:
        l1.append(i-4)
    else:
        l1.append(-1)
    moved.append(l1)
# Part A
for i in range(16):
    l1 = moved[i]
    k = 0
    for b1 in l1:
        l = 0
        for r in l1:
            if r != -1 and b1 != -1:
                for b2 in range(16):
                    x1 = int(number_encoded[b1, b2, r, 0])
                    x2 = int(number_encoded[i, b2, i, 0])
                    transitions[b1, b2, r, 0, k, i, b2, i, 0] = (0.5 - p)*trans_r[b1, b2, r, 0, l]
                    encoded_transitions[x1, k, x2] = transitions[b1, b2, r, 0, k, i, b2, i, 0]
            l = l + 1
        k = k + 1
    k = 4
    for b2 in l1:
        l = 0
        for r in l1:
            if r != -1 and b2 != -1:
                for b1 in range(16):
                    transitions[b1, b2, r, 1, k, b1, i, i, 1] = (0.5 - p)*trans_r[b1, b2, r, 1, l]
                    x1 = int(number_encoded[b1, b2, r, 1])
                    x2 = int(number_encoded[b1, i, i, 1])
                    encoded_transitions[x1, k, x2] = transitions[b1, b2, r, 1, k, b1, i, i, 1]
            l = l + 1
        k = k + 1
# Part B
for i in range(16):
    if i < 15:
        for b2 in range(16):
            transitions[i, b2, i+1, 0, 1, i+1, b2, i,0] = (0.5 - p)*trans_r[i, b2, i+1, 0, 0]
            x1 = int(number_encoded[i, b2, i+1, 0])
            x2 = int(number_encoded[i+1, b2, i, 0])
            encoded_transitions[x1, 1, x2] = transitions[i, b2, i+1, 0, 1, i+1, b2, i, 0]
        for b1 in range(16):
            transitions[b1, i, i+1, 1, 5, b1, i+1, i, 1] = (0.5 - p)*trans_r[b1, i, i+1, 1, 0]
            x1 = int(number_encoded[b1, i, i+1, 1])
            x2 = int(number_encoded[b1, i+1, i, 1])
            encoded_transitions[x1, 5, x2] = transitions[b1, i, i+1, 1, 5, b1, i+1, i, 1]

    if i > 0:
        for b2 in range(16):
            transitions[i, b2, i-1, 0, 0, i-1, b2, i, 0] = (0.5 - p)*trans_r[i, b2, i-1, 0, 1]
            x1 = int(number_encoded[i, b2, i-1, 0])
            x2 = int(number_encoded[i-1, b2, i, 0])
            encoded_transitions[x1, 0, x2] = transitions[i, b2, i-1, 0, 0, i-1, b2, i, 0]

        for b1 in range(16):
            transitions[b1, i, i-1, 1, 4, b1, i-1, i, 1] = (0.5 - p)*trans_r[b1, i, i-1, 1, 1]
            x1 = int(number_encoded[b1, i, i-1, 1])
            x2 = int(number_encoded[b1, i-1, i, 1])
            encoded_transitions[x1, 4, x2] = transitions[b1, i, i-1, 1, 4, b1, i-1, i, 0]

    if i > 3:
        for b2 in range(16):
            transitions[i, b2, i-4, 0, 2, i-4, b2, i, 0] = (0.5 - p)*trans_r[i, b2, i-4, 0, 3]
            x1 = int(number_encoded[i, b2, i-4, 0])
            x2 = int(number_encoded[i-4, b2, i, 0])
            encoded_transitions[x1, 2, x2] = transitions[i, b2, i-4, 0, 2, i-4, b2, i, 0]
        for b1 in range(16):
            transitions[b1, i, i-4, 1, 6, b1, i-4, i, 1] = (0.5 - p)*trans_r[b1, i, i-4, 1, 3]
            x1 = int(number_encoded[b1, i, i-4, 1])
            x2 = int(number_encoded[b1, i-4, i, 1])
            encoded_transitions[x1, 6, x2] = transitions[b1, i, i-4, 1, 6, b1, i-4, i, 1]

    if i < 12:
        for b2 in range(16):
            transitions[i, b2, i+4, 0, 3, i+4, b2, i, 0] = (0.5 - p)*trans_r[i, b2, i+4, 0, 2]
            x1 = int(number_encoded[i, b2, i+4, 0])
            x2 = int(number_encoded[i+4, b2, i, 0])
            encoded_transitions[x1, 3, x2] = transitions[i, b2, i+4, 0, 3, i+4, b2, i, 0]
        for b1 in range(16):
            transitions[b1, i, i+4, 1, 7, b1, i+4, i, 1] = (0.5 - p)*trans_r[b1, i, i+4, 1, 2]
            x1 = int(number_encoded[b1, i, i+4, 1])
            x2 = int(number_encoded[b1, i+4, i, 1])
            encoded_transitions[x1, 7, x2] = transitions[b1, i, i+4, 1, 7, b1, i+4, i, 1]


# Passing.

def get(position):
    return [position % 4, int(position/4)]


coordinates = []
for i in range(16):
    coordinates.append(get(i))


def possible_positions(i1, i2):
    p1 = coordinates[i1]
    p2 = coordinates[i2]
    x1 = min(p1[0], p2[0])
    y1 = min(p1[1], p2[1])
    x2 = max(p1[0], p2[0])
    y2 = max(p1[1], p2[1])
    if x1 == x2:
        l = []
        for y in range(y1, y2+1):
            l.append(x1+4*y)
        return np.array(l, dtype=int), max(x2-x1, y2-y1)
    elif y1 == y2:
        l = []
        for x in range(x1, x2+1):
            l.append(x+4*y1)
        return np.array(l, dtype=int), max(x2-x1, y2-y1)
    elif p1[0] + p1[1] == p2[0] + p2[1]:
    	l = []
    	k = p1[0] + p1[1]
    	for x in range(x1, x2 + 1):
    		l.append(x + 4*(k-x))
    	return l, max(x2-x1, y2-y1)
    elif p1[0] - p1[1] == p2[0] - p2[1]:
    	l = []
    	k = p1[0] - p1[1]
    	for x in range(x1, x2+1):
    		l.append(x + 4*(x-k))
    	return l, max(x2-x1, y2-y1)
    return [-1], max(x2-x1, y2-y1)


for b1 in range(16):
    for b2 in range(16):
        R, v = possible_positions(b1, b2)
        for poss in range(2):
            for r in range(16):
                x1 = int(number_encoded[b1, b2, r, poss])
                if r < 15:
                    transitions[b1, b2, r, poss, 8, b1, b2, r+1, 1-poss] = (q-0.1*v)
                    if r+1 in R:
                        transitions[b1, b2, r, poss, 8, b1, b2, r+1, 1 -poss] = transitions[b1, b2, r, poss, 8, b1, b2, r+1, 1-poss]/2
                    x2 = int(number_encoded[b1, b2, r+1, 1-poss])
                    encoded_transitions[x1, 8, x2] = transitions[b1, b2, r, poss, 8, b1, b2, r+1, 1-poss]*trans_r[b1, b2, r, poss, 1]
                    transitions[b1, b2, r, poss, 8, b1, b2, r+1, 1-poss] = transitions[b1, b2, r, poss, 8, b1, b2, r+1, 1-poss] * trans_r[b1, b2, r, poss, 1]
                if r > 0:
                    transitions[b1, b2, r, poss, 8, b1, b2, r-1, 1-poss] = (q-0.1*v)
                    if r-1 in R:
                        transitions[b1, b2, r, poss, 8, b1, b2, r-1, 1 -poss] = transitions[b1, b2, r, poss, 8, b1, b2, r-1, 1-poss]/2
                    x2 = int(number_encoded[b1, b2, r-1, 1-poss])
                    encoded_transitions[x1, 8, x2] = transitions[b1, b2, r, poss, 8, b1, b2, r-1, 1-poss]*trans_r[b1, b2, r, poss, 0]
                    transitions[b1, b2, r, poss, 8, b1, b2, r-1, 1-poss] = transitions[b1, b2, r, poss, 8, b1, b2, r-1, 1-poss] * trans_r[b1, b2, r, poss, 0]
                if r > 3:
                    transitions[b1, b2, r, poss, 8, b1, b2, r-4, 1-poss] = (q-0.1*v)
                    if r-4 in R:
                        transitions[b1, b2, r, poss, 8, b1, b2, r-4, 1 -poss] = transitions[b1, b2, r, poss, 8, b1, b2, r-4, 1-poss]/2
                    x2 = int(number_encoded[b1, b2, r-4, 1-poss])
                    encoded_transitions[x1, 8, x2] = transitions[b1, b2, r, poss, 8, b1, b2, r-4, 1-poss] * trans_r[b1, b2, r, poss, 2]
                    transitions[b1, b2, r, poss, 8, b1, b2, r-4, 1-poss] = transitions[b1, b2, r, poss, 8, b1, b2, r-4, 1-poss] * trans_r[b1, b2, r, poss, 2]
                if r < 12:
                    transitions[b1, b2, r, poss, 8, b1, b2, r+4, 1-poss] = (q-0.1*v)
                    if r+4 in R:
                        transitions[b1, b2, r, poss, 8, b1, b2, r+4, 1 -poss] = transitions[b1, b2, r, poss, 8, b1, b2, r+4, 1-poss]/2
                    x2 = int(number_encoded[b1, b2, r+4, 1-poss])
                    encoded_transitions[x1, 8, x2] = transitions[b1, b2, r, poss, 8, b1, b2, r+4, 1-poss] * trans_r[b1, b2, r, poss, 3]
                    transitions[b1, b2, r, poss, 8, b1, b2, r+4, 1-poss] = transitions[b1, b2, r, poss, 8, b1, b2, r+4, 1-poss] * trans_r[b1, b2, r, poss, 3]

# Convert into states now.


# Shooting

for b1 in range(16):
    for b2 in range(16):
        x = [get(b1)[0], get(b2)[0]]
        for poss in range(2):
            for r in range(16):
                x1 = int(number_encoded[b1, b2, r, poss])
                t_value = 0
                if r < 15:
                    value = q - 0.2*(3-x[poss])
                    if r+1 == 7 or r + 1 == 11:
                        value = value/2
                    t_value += ((value) * trans_r[b1, b2, r, poss, 1])
                if r > 0:
                    value = q - 0.2*(3-x[poss])
                    if r-1 == 7 or r-1 == 11:
                        value = value/2
                    t_value += ((value) * trans_r[b1, b2, r, poss, 0])
                if r > 3:
                    value = q - 0.2*(3-x[poss])
                    if r-4 == 7 or r-4 == 11:
                        value = value/2
                    t_value += ((value) * trans_r[b1, b2, r, poss, 2])
                if r < 12:
                    value = q - 0.2*(3-x[poss])
                    if r+4 == 7 or r+4 == 11:
                        value = value/2
                    t_value += ((value) * trans_r[b1, b2, r, poss, 3])
                encoded_transitions[x1, 9, 8192] = t_value


reward = np.zeros((16*16*16*2+1, 10, 16*16*16*2+1))
reward[:8192, 9, 8192] = 1

print("numStates", 16*16*16*2+1)
print("numActions", 10)
print("end", 8192)
encoded_transitions = np.around(encoded_transitions, decimals=6)
for i in range(8193):
    for j in range(8193):
        for k in range(10):
            if encoded_transitions[i, k, j] != 0:
                print("transition", i, k, j, reward[i, k, j], encoded_transitions[i, k, j])

print("mdptype", "episodic")
print("discount", 1)


# Setting the reward.
