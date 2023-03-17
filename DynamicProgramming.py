#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        a = np.argmax(self.Q_sa[s])
        return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        sum_updates = p_sas[s,a] * (r_sas[s,a] + self.gamma * np.max(self.Q_sa, -1))
        self.Q_sa[s,a] = np.sum(sum_updates)
    
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)    
        
    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
    i = 0
    max_error = np.inf
    while (max_error >= threshold):
        i += 1
        max_error = 0
        for s in range(0,QIagent.n_states):
            for a in range(0,QIagent.n_actions):
                x = QIagent.Q_sa[s,a]
                QIagent.update(s, a, env.p_sas, env.r_sas)
                max_error = np.max([max_error, abs(x - QIagent.Q_sa[s,a])])

        # Plot current Q-value estimates & print max error
        # env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)
        # print("Q-value iteration, iteration {}, max error {}".format(i, max_error))

    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    # View optimal policy
    done = False
    s = env.reset()
    rewards = []
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        rewards.append(r)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
        s = s_next

    # Compute mean reward per timestep under the optimal policy
    mean_reward_per_timestep = np.mean(np.array(rewards))
    # print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    # return mean_reward_per_timestep

if __name__ == '__main__':
    experiment()

    # mean_r = []
    # for i in range(0,1000):
    #     m_r = experiment()
    #     mean_r.append(m_r)
    # print(np.mean(np.array(mean_r)))
    
