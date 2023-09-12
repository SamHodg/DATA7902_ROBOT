import sys
import time
from constants import *
from environment import *
from state import State
import random
import math

"""
solution.py

"""


class RLAgent:

    #
    # TODO: (optional) Define any constants you require here.
    #

    def __init__(self, environment: Environment):
        self.environment = environment
        #
        # TODO: (optional) Define any class instance variables you require (e.g. Q-value tables) here.
        #
        self.q_values = {} # Could set to 0 for initial state
        #self.current_state = self.environment.get_init_state()
        self.epsilon_start = 1 # Play around with this value!
        self.epsilon_final = 0.001
        self.epsilon_decay = 1000
        self.alpha = 0.1

# NEW STUFF        
        self.collision_ahead = False
        self.collision_behind = False
        
        pass

    # === Q-learning ===================================================================================================

    def q_learn_train(self):
        """
        Train this RL agent via Q-Learning.
        """
        #
        # TODO: Implement your Q-learning training loop here.
        #
        # Conditions:
        under_train_threshold = True # Training cost not reached threshold
        episode_costs = []
        last_Z_avg_episode_costs = []
        Y = 50 # For following condition
        last_Y_avg_not_satisfied = True # Average last Y episodes below reward target threshold
        X, Z = (1000,1000) # For following condition
        last_X_contains_max_Z_avg = True # Last X episodes contains max average of last Z rewards
        num_episodes = 0
        R50List = []
        episode_number = []
        #while under_train_threshold and last_Y_avg_not_satisfied and last_X_contains_max_Z_avg:
        #while under_train_threshold and last_Y_avg_not_satisfied:
        while num_episodes < 1:  # Was 2500
            self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1 * num_episodes / self.epsilon_decay)
            #self.epsilon = 0.15
            # Start from initial state
            state = self.environment.get_init_state()
            # Complete an episode
            max_steps = 300
            steps = 0
# NEW STUFF
            visited_states = []
            Currently_biased = {}
            # TUNE THIS BIAS VALUE
            Bias = {}
            # Accuracy of image classification (test for varying levels)
            ImAcc = 100
            
            while not self.environment.is_solved(state) and steps <= max_steps:
# NEW STUFF
                if state not in visited_states:
                                        
                    # Get coordinates ahead
                    fr, fc = self.get_forward_cell_coords(state.robot_posit, state.robot_orient)
                    # Get coordinates behind
                    br, bc = self.get_rear_cell_coords(state.robot_posit, state.robot_orient)
                    
                    # Check ahead for out of bounds, collision with obstacle and collision with hazard
                    if (not 0 <= fr < self.environment.n_rows) or (not 0 <= fc < self.environment.n_cols) or self.environment.obstacle_map[fr][fc] or self.environment.hazard_map[fr][fc]:
                        # Collision ahead
                        self.collision_ahead = True
                    else:
                        # No collision ahead
                        self.collision_ahead = False
                        
                    # Check behind for out of bounds, collision with obstacle and collision with hazard
                    if (not 0 <= br < self.environment.n_rows) or (not 0 <= bc < self.environment.n_cols) or self.environment.obstacle_map[br][bc] or self.environment.hazard_map[br][bc]:
                        # Collision behind
                        self.collision_behind = True
                    else:
                        # No collision behind
                        self.collision_behind = False

# HERE - FIX BIASES (see notes)

                    # Apply image classification accuracy for FORWARD
                    # RD = random.uniform(0,100)
                    
                    # # BINARY - CAN IMPROVE ON THIS
                    # if self.collision_ahead:  # Collision ahead
                    #     if RD <= ImAcc:  # Image classified correctly
                    #         # Correctly apply negative bias
                    #         self.q_values[(state, FORWARD)] = self.q_values.get((state, FORWARD), 0) - Bias
                    #         forward_bias_is = 'Neg'
                    #     else:  # Image classified incorrectly
                    #         # Incorrectly apply positive bias
                    #         self.q_values[(state, FORWARD)] = self.q_values.get((state, FORWARD), 0) + Bias 
                    #         forward_bias_is = 'Pos'
                    # else:  # No collision ahead
                    #     if RD <= ImAcc:  # Image classified correctly
                    #         # Correctly apply positive bias
                    #         self.q_values[(state, FORWARD)] = self.q_values.get((state, FORWARD), 0) + Bias
                    #         forward_bias_is = 'Pos'
                    #     else:  # Image classified incorrectly
                    #         # Incorrectly apply negative bias
                    #         self.q_values[(state, FORWARD)] = self.q_values.get((state, FORWARD), 0) - Bias
                    #         forward_bias_is = 'Neg'
                    
                    # Apply image classification accuracy
                    RD_fwd = random.uniform(0,1)
                    RD_bck = random.uniform(0,1)
                    # Cube root
                    RD_fwd = RD_fwd**(1/3)
                    RD_bck = RD_bck**(1/3)
                    
                    if self.collision_ahead:  # Collision ahead
                        # Negatively weight
                        RD_fwd = 1 - RD_fwd                      
                    if self.collision_behind:  # Collision behind
                        # Negatively weight
                        RD_bck = 1 - RD_bck
                      
                    # Shift range to [-0.5, 0.5]
                    RD_fwd = RD_fwd - 0.5
                    RD_bck = RD_bck - 0.5
                    
                    # Bias modifier (tune this)
                    BiasMod = 10
                    # Assign bias
                    Bias[(state,FORWARD)] = RD_fwd * BiasMod
                    Bias[(state,REVERSE)] = RD_bck * BiasMod
                    
                    # Apply bias to Q-value
                    self.q_values[(state, FORWARD)] = self.q_values.get((state, FORWARD), 0) + Bias[(state,FORWARD)]
                    self.q_values[(state, REVERSE)] = self.q_values.get((state, REVERSE), 0) + Bias[(state,REVERSE)]
                               
                    Currently_biased[(state, FORWARD)] = True
                    Currently_biased[(state, REVERSE)] = True
                    visited_states.append(state)
                    
                action = self.q_learn_select_action(state)
                reward, next_state = self.environment.perform_action(state, action)
                # Update q-value for the (state, action) pair
                old_q = self.q_values.get((state, action), 0)
                best_next = self.get_best_q_action(next_state)
                best_next_q = self.q_values.get((next_state, best_next), 0)
                if self.environment.is_solved(next_state):  # Next state is a target
                    best_next_q = 0
                target = reward + self.environment.gamma * best_next_q
                
                new_q = old_q + self.alpha * (target - old_q)
                self.q_values[(state, action)] = new_q

# NEW STUFF
                if action in [FORWARD, REVERSE] and Currently_biased[(state, action)] and self.environment.move_equals_action:
                   #  # BINARY - IMPROVED UPON
                   #  # Remove bias after actually performing action
                   # if action == FORWARD:
                   #     if forward_bias_is == 'Pos':
                   #         self.q_values[(state, action)] += - Bias
                   #     else:
                   #         self.q_values[(state, action)] += Bias
                   # else:
                   #     if backward_bias_is == 'Pos':
                   #         self.q_values[(state, action)] += - Bias
                   #     else:
                   #         self.q_values[(state, action)] += Bias
                   
                   # Remove bias
                   self.q_values[(state, action)] -= Bias[(state,action)]
                    # No longer biased
                   Currently_biased[(state, action)] = False
                
                # Update state
                state = next_state
                
                steps += 1
            
            num_episodes += 1
            
            total_reward = self.environment.get_total_reward()
            # Check if we've reached training cost threshold
            if total_reward <=  self.environment.training_reward_tgt:
                under_train_threshold = False
#                print("under_train_treshold violated")
                
            # Update episode costs
            episode_costs.append(total_reward-sum(episode_costs))

            # Investigate last_Y_avg_not_satisfied
            if len(episode_costs) >= Y:
                # Check condition
                R50 = sum(episode_costs[-Y:])/Y
                R50List.append(R50)
                episode_number.append(num_episodes)
                if num_episodes == 10000 or num_episodes == 25000 or num_episodes == 40000:
                    print(num_episodes)
                if R50 >= self.environment.evaluation_reward_tgt:
                    last_Y_avg_not_satisfied = False
#                    print("last_Y_avg_not_satisfied violated")
                
            # Investigate last_X_contains_max_Z_avg
            if len(episode_costs) >= Z:
                last_Z_avg_episode_costs.append(sum(episode_costs[-Z:])/Z)
                # Last episode had highest reward so far
                if max(last_Z_avg_episode_costs) == last_Z_avg_episode_costs[-1]:
                    best_policy = self.q_values
                    best_iteration = num_episodes
                # Need least X+Y entries
                if len(last_Z_avg_episode_costs) >= X:
                    # Max last-Z-average not in last X episodes
                    if max(last_Z_avg_episode_costs) > max(last_Z_avg_episode_costs[-X:]):
                        last_X_contains_max_Z_avg = False
                        # return to best discovered policy
                        self.q_values = best_policy
#                        print("Best episode: ",best_iteration)
#                        print("last_X_contains_max_Z_avg violated")
#                        print("Length = ",len(last_Z_avg_episode_costs)+Z)  
#                        print("old max = ",max(last_Z_avg_episode_costs),"new max = ",max(last_Z_avg_episode_costs[-X:]))
        print("Episodes: ",num_episodes)
        f = open("Qlearn4vals.txt", "w")
        f.write(str(R50List))
        f = open("Qlearn4eps.txt", "w")
        f.write(str(episode_number))
        pass

    def q_learn_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via Q-learning.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        #
        # TODO: Implement code to return an approximately optimal action for the given state (based on your learned
        #  Q-learning Q-values) here.
        #
        # Using epsilon-greedy
        best_a = self.get_best_q_action(state)
        if best_a is None or random.random() < self.epsilon:
            return random.choice(ROBOT_ACTIONS)
        return best_a        
        
        pass

    # === SARSA ========================================================================================================

    def sarsa_train(self):
        """
        Train this RL agent via SARSA.
        """
        #
        # TODO: Implement your SARSA training loop here.
        #
        # Conditions:
        under_train_threshold = True # Training cost not reached threshold
        episode_costs = []
        last_Z_avg_episode_costs = []
        Y = 100 # For following condition
        last_Y_avg_not_satisfied = True # Average last Y episodes below reward target threshold
        X, Z = (1000,1000) # For following condition
        last_X_contains_max_Z_avg = True # Last X episodes contains max average of last Z rewards
        num_episodes = 0
        R50List = []
        episode_number = []
        #while under_train_threshold and last_Y_avg_not_satisfied and last_X_contains_max_Z_avg:
        #while under_train_threshold and last_Y_avg_not_satisfied:
        while num_episodes < 2500:
            self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1 * num_episodes / self.epsilon_decay)
            # Start from initial state
            state = self.environment.get_init_state()
            # Starting action (unique to SARSA)
            action = self.sarsa_select_action(state)
            # Complete an episode
            max_steps = 300
            steps = 0
            while not self.environment.is_solved(state) and steps <= max_steps:
                reward, next_state = self.environment.perform_action(state, action)
                next_action = self.sarsa_select_action(next_state)
                # Update q-value for the (state, action) pair
                old_q = self.q_values.get((state, action), 0)
                next_q = self.q_values.get((next_state, next_action), 0)
                if self.environment.is_solved(next_state):  # Next state is a target
                    next_q = 0
                target = reward + self.environment.gamma * next_q
                
                #new_q = old_q + self.environment.alpha*2 * (target - old_q)
                new_q = old_q + self.alpha * (target - old_q)
                self.q_values[(state, action)] = new_q
                
                # Update state and action
                state = next_state
                action = next_action
                
                steps += 1
            
            num_episodes += 1
            total_reward = self.environment.get_total_reward()
            # Check if we've reached training cost threshold
            if total_reward <=  self.environment.training_reward_tgt:
                under_train_threshold = False
                #print("under_train_treshold violated")
                
            # Update episode costs
            episode_costs.append(total_reward-sum(episode_costs))

            # Investigate last_Y_avg_not_satisfied
            if len(episode_costs) >= Y:
                # Check condition
                R50 = sum(episode_costs[-Y:])/Y
                R50List.append(R50)
                episode_number.append(num_episodes)
                if R50 >= self.environment.evaluation_reward_tgt:
                    last_Y_avg_not_satisfied = False
                    #print("last_Y_avg_not_satisfied violated")
                
            # Investigate last_X_contains_max_Z_avg
            if len(episode_costs) >= Z:
                last_Z_avg_episode_costs.append(sum(episode_costs[-Z:])/Z)
                # Last episode had highest reward so far
                if max(last_Z_avg_episode_costs) == last_Z_avg_episode_costs[-1]:
                    best_policy = self.q_values
                # Need least X+Y entries
                if len(last_Z_avg_episode_costs) >= X:
                    # Max last-Z-average not in last X episodes
                    if max(last_Z_avg_episode_costs) > max(last_Z_avg_episode_costs[-X:]):
                        last_X_contains_max_Z_avg = False
                        # return to best discovered policy
                        self.q_values = best_policy
                        #print("last_X_contains_max_Z_avg violated")
                        #print("Length = ",len(last_Z_avg_episode_costs)+Z)  
                        #print("old max = ",max(last_Z_avg_episode_costs),"new max = ",max(last_Z_avg_episode_costs[-X:]))
        print("Episodes: ",num_episodes)
        f = open("SARSA4vals.txt", "w")
        f.write(str(R50List))
        f = open("SARSA4eps.txt", "w")
        f.write(str(episode_number))
        pass

    def sarsa_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via SARSA.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        #
        # TODO: Implement code to return an approximately optimal action for the given state (based on your learned
        #  SARSA Q-values) here.
        #
         # Using epsilon-greedy
        best_a = self.get_best_q_action(state)
        if best_a is None or random.random() < self.epsilon:
            return random.choice(ROBOT_ACTIONS)
        return best_a        
             
        pass

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: (optional) Add any additional methods here.
    #
    # Note: self.environment.gamma = discount rate
    # self.environment.alpha = learning rate
    
    # Currently starting from initial state every time, could try expanding state set
    # as we go then randomly drawing from state set to start (set better than list?)
    
    def get_best_q_action(self, state: State):
        best_q = float('-inf')
        best_a = None
        for action in ROBOT_ACTIONS:
            this_q = self.q_values.get((state, action))
            if this_q is not None and this_q > best_q:
                best_q = this_q
                best_a = action
        return best_a
    
# NEW STUFF    
    def get_forward_cell_coords(self, posit, direction):
        """
        Return the coordinates of the cell adjacent to the given position in the given direction.
        orientation.
        :param posit: position
        :param direction: direction (element of ROBOT_ORIENTATIONS)
        :return: (row, col) of adjacent cell
        """
        r, c = posit
        if direction == ROBOT_UP:
            return r - 1, c
        elif direction == ROBOT_DOWN:
            return r + 1, c
        elif direction == ROBOT_UP_LEFT:
            if c % 2 == 0:
                return r - 1, c - 1
            else:
                return r, c - 1
        elif direction == ROBOT_UP_RIGHT:
            if c % 2 == 0:
                return r - 1, c + 1
            else:
                return r, c + 1
        elif direction == ROBOT_DOWN_LEFT:
            if c % 2 == 0:
                return r, c - 1
            else:
                return r + 1, c - 1
        else:   # direction == ROBOT_DOWN_RIGHT
            if c % 2 == 0:
                return r, c + 1
            else:
                return r + 1, c + 1

    def get_rear_cell_coords(self, posit, direction):
        """
        Return the coordinates of the cell BEHIND the given position in the given
        direction orientation.
        :param posit: position
        :param direction: direction (element of ROBOT_ORIENTATIONS)
        :return: (row, col) of adjacent cell
        """
        r, c = posit
        if direction == ROBOT_UP:
            return r + 1, c
        elif direction == ROBOT_DOWN:
            return r - 1, c
        elif direction == ROBOT_UP_LEFT:
            if c % 2 == 0:
                return r, c + 1
            else:
                return r + 1, c + 1
        elif direction == ROBOT_UP_RIGHT:
            if c % 2 == 0:
                return r, c - 1
            else:
                return r + 1, c - 1
        elif direction == ROBOT_DOWN_LEFT:
            if c % 2 == 0:
                return r - 1, c + 1
            else:
                return r, c + 1
        else:   # direction == ROBOT_DOWN_RIGHT
            if c % 2 == 0:
                return r - 1, c - 1
            else:
                return r, c - 1
    
    

# Q-value of STATE-ACTION PAIR, not just state
# MAINTAIN A LIST OF VISITED STATES (position AND ORIENTATION)
# If state not visited before, apply observation bias

# Ensure blind and informed algorithms have same probability seeds
# 
# TWO CAMERAS: FRONT AND BACK
# Apply oservation bias to forward and reverse actions.
# Probably don't apply observation bias to turning actions
# If nominal
# 
# For reporting purposes, track total "collisions" and measure reduction
# (As well as overall cost reduction)
