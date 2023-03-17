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

class MonteCarloAgent:

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
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # Start reward sumation from 0
        T_ep = len(actions)
        G_t = np.zeros(T_ep + 1)
        i = T_ep - 1
        while i >= 0:
            # Compute Monte Carlo target at each step
            G_t[i] = rewards[i] + self.gamma*G_t[i+1]

            # Update Q-table
            self.Q_sa[states[i],actions[i]] += self.learning_rate * (G_t[i] - self.Q_sa[states[i],actions[i]])

            i -= 1


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []
    
    t = 0
    while (t < n_timesteps):
        # Initialize Episode
        s = env.reset()
        actions = []
        states = [s]
        rew = []
        
        # Collect Episode
        for i in range(0, max_episode_length):
            # Add budget
            t += 1

            # Apply Monte Carlo
            actions.append(pi.select_action(s, policy, epsilon, temp))
            s_next, r, done = env.step(actions[i])
            states.append(s_next)
            rew.append(r)
            rewards.append(r)
            s = s_next
            if (done) or (t >= n_timesteps):
                break
        
        # Update using Monte Carlo RL
        pi.update(states, actions, rew)
        
        # Plot the Q-value estimates during Monte Carlo RL execution
        # if plot:
        #     env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1)
        # t += 1
    
    return rewards 
    
def test():
    n_timesteps = 10
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format((rewards)))  
            
if __name__ == '__main__':
    test()
