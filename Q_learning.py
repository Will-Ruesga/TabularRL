#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

np.random.seed(123)

class QLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
            
            # Epsilon-Greedy policy
            size = len(self.Q_sa[s])
            probs = np.array([epsilon / size] * size)
            probs[argmax(self.Q_sa[s])] = 1 - (epsilon * ((size - 1) / size))

            a = np.random.choice(self.n_actions, p=probs)
            
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
            
            # Boltzmann policy
            a = np.random.choice(self.n_actions, p=softmax(self.Q_sa[s], temp))
        return a
        
    def update(self,s ,a ,r ,s_next, done):
        G_t = r + self.gamma * np.max(self.Q_sa[s_next], -1)
        self.Q_sa[s,a] += self.learning_rate * (G_t - self.Q_sa[s,a])
        

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=True)
    rewards = []

    # Initialize Q-Learning Agent and Initial State
    pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    s = env.reset()

    t = 0
    while (t < n_timesteps):        
        a = pi.select_action(s, policy, epsilon, temp)

        # Simulate evniorment
        s_next, r, done = env.step(a)
        rewards.append(r)
        pi.update(s, a, r, s_next, done)

        # Reset enviorment
        if done:
            s = env.reset()
        else:
            s = s_next

        # Plot the Q-value estimates during Q-learning execution
        # if plot:
        #     env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1)
        t += 1
    return rewards 

def test():
    
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(np.max(rewards)))

if __name__ == '__main__':
    test()
