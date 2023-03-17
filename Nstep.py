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

# np.random.seed(123)

class NstepQLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, n):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n = n
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
        
    def update(self, states, actions, rewards, done):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        T_ep = len(actions)
        for t in range(0, T_ep):
            # Compute numer of rewards left to sum
            m = np.min([self.n, (T_ep - t)])

            # N-step whithout bootstrap
            G_t = 0
            for i in range(0, m):
                G_t += self.gamma**i * rewards[i]
            # Add boostrap if needed
            if (not done) or (t+m < T_ep):
                G_t += self.gamma**m * np.max(self.Q_sa[states[t+m]])
            
            # Update Q-table
            self.Q_sa[states[t],actions[t]] += self.learning_rate * (G_t - self.Q_sa[states[t],actions[t]])
        

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)
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

            # Apply N-Step
            actions.append(pi.select_action(s, policy, epsilon, temp))
            s_next, r, done = env.step(actions[i])
            states.append(s_next)
            rew.append(r)
            rewards.append(r)
            s = s_next
            if (done) or (t >= n_timesteps):
                break
            
        # Compute n-step targets and update
        pi.update(states, actions, rew, done)
        # t += 1
    
    return rewards

def test():
    n_timesteps = 50000
    max_episode_length = 10000
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    print("Obtained rewards: {}".format(np.max(rewards)))    
    
if __name__ == '__main__':
    test()
