import os
import sys
import random 
import json
import math
import utils
import time
import config
import numpy
random.seed(72)

def abtuse_angle(point, point1, point2):
    # To know whether the angle is abtuse or acute.
    #Done
    vector1 = (point1[0] - point[0], point1[1] - point[1])
    vector2 = (point2[0] - point[0], point2[1] - point[1])
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    return dot_product < 0

def distance(x1, y1, x2, y2):
    # to calculate the distance between two points.
    return (x1-x2)**2 + (y1 - y2)**2 

def state_calculate(value):
    #Distance of the point from nearest hole and the corresponding coordinates. 
    x, y = value[0], value[1]
    if x <= 250:
        if y <= 250:
            return distance(40, 40, x, y), (40, 40)
        elif y <= 500:
            return distance(40, 460, x, y), (40, 460)
    elif x <= 750:
        if y <= 250:
            return distance(500, 40, x, y), (500, 40)
        elif y <= 500:
            return distance(500, 460, x, y), (500, 460)
    elif x <= 1000:
        if y <= 250:
            return distance(960, 40, x, y), (960, 40)
        elif y <= 500:
            return distance(960, 460, x, y), (960, 460)

def find_point_on_line(value1, value2, k):
    #find the point where the cue has to hit the ball.
    #Done
    x1 = value1[0]
    y1 = value1[1]
    x2 = value2[0]
    y2 = value2[1]
    dx = x1 - x2
    dy = y1 - y2

    magnitude = math.sqrt(dx**2 + dy**2)

    unit_vector = (dx / magnitude, dy / magnitude)

    x3 = x1 + unit_vector[0] * k
    y3 = y1 + unit_vector[1] * k

    return (x3, y3)

def angle_calculate(position1, position2):
    #angle of the line 
    #Done
    x1 = position1[0]
    y1 = position1[1]
    x2 = position2[0]
    y2 = position2[1]

    dx = x2 - x1
    dy = y2 - y1
    angle = math.atan2(-dy, dx)
    angle = angle/math.pi
    if( angle >= 0 and angle <= 0.5):
        angle = -(0.5-angle)
    elif( angle >= 0.5 and angle <= 1):
        angle = angle-0.5
    elif( angle <= 0 and angle >= -0.5):
        angle = -(0.5-angle)
    else:
        angle = 1.5 + angle
    return angle            

        
class Agent:
    def __init__(self, table_config) -> None:
        self.table_config = table_config
        self.prev_action = None
        self.curr_iter = 0
        self.state_dict = {}
        self.holes =[]
        self.ns = utils.NextState()
        self.ball_radius = -1

    def set_holes(self, holes_x, holes_y, radius):
        for x in holes_x:
            for y in holes_y:
                self.holes.append((x[0], y[0]))
        self.ball_radius = radius

    def evaluation(self, ball_pos,  next_state):
        #Get the score.
        score = 0
        changed = False
        for x in ball_pos.keys():
            if x != 0 and x != "white":
                if x not in next_state.keys():
                    score += 10
                    changed = True
                else:
                    d1, s1 = state_calculate(ball_pos[x])
                    d2, s2 = state_calculate(next_state[x])
                    if( d2 < d1):
                        changed = True                
                        score += (0.00001 * (d1-d2))
        if( not changed ):
            score = -0.02  
        #print(score) 
        return score
    def ball_choice(self, ball_pos):
        #Choosing the ball.
        sorted_key = {}
        for key in ball_pos.keys():
            if key != 0 and key != "white":
                state_final = -1
                ball_position = ball_pos[key]
                distance_final = float('inf')
                minimum_distance_state = None
                minimum_distance = float('inf')
                for pos in self.holes:
                    abtuseangle = abtuse_angle(ball_position, ball_pos[0], pos)  
                    gap = distance(ball_position[0], ball_position[1], pos[0], pos[1])
                    if( abtuseangle):
                        if(distance_final > gap):
                            distance_final = gap
                            state_final = pos  
                    else:
                        if( minimum_distance > gap):  
                            minimum_distance = gap
                            minimum_distance_state = pos
                if( state_final != -1):
                    sorted_key[key] = [distance_final, state_final]
                else:
                    sorted_key[key] =  [minimum_distance, minimum_distance_state]
        sorted_key = dict(sorted(sorted_key.items(), key=lambda item: item[1][0]))
        return sorted_key

    def action(self, ball_pos=None): 
        #Which action to take.
        best_score = -2
        best_angle = None
        best_force = None
        sorted_key = self.ball_choice(ball_pos)
        keys = list(sorted_key.keys())
        for i in range(5):
            ball_position = ball_pos[keys[i%len(keys)]]
            state_i = sorted_key[keys[i%len(keys)]][1]
            abtuseangle = abtuse_angle(ball_position, ball_pos[0], state_i)                
            if(not abtuseangle):
                dist = 0
            else:
                dist = 2*self.ball_radius
            new_point = find_point_on_line(ball_position, state_i, dist)
            angle_max = angle_calculate(ball_pos[0], new_point)
            for j in range(4):
                Force = random.uniform(0, 1)
                next_state = self.ns.get_next_state(ball_pos, (angle_max, Force), 20)
                score = self.evaluation(ball_pos, next_state)
                if( score > best_score):
                    best_angle = angle_max
                    best_force = Force
                    best_score = score
        return ((best_angle, best_force))

