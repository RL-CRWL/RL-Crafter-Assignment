'''
Components of a PPO Algorithm:
Actor: This is the policy network responsible for choosing actions to take in a given state. 
Critic: This is the value function approximator that estimates the value of states or state-action pairs, 
providing feedback to the actor. 
Generalized Advantage Estimation (GAE): A technique used by PPO to compute advantage values, 
which helps reduce variance in policy gradient estimates. 
Clipped Surrogate Objective: The core mechanism of PPO that limits the magnitude of policy updates during training, ensuring stability. 
'''

