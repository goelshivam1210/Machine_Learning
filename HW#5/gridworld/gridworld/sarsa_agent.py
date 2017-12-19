import grid_environment as env
import numpy as np
import math
import random
import copy
import utils
import csv


class Sarsa_Agent:
	def __init__(self, numStates, numActions, epsilon, gamma, alpha, init_value):
		self.numStates = numStates
		self.numActions = numActions
		self.epsilon = epsilon
		self.gamma = gamma
		self.alpha = alpha
		self.init_value = init_value
		self.accumulated_reward = 0
		self.track = []
		self.numstep = 0
		self.battery_max = 5

		#init array Q
		self.Q_value = None
		self.behavior_value = None
		self.history_max = None 

		self.last_action = None
		self.last_observation = None

		self.Q_init()


	def Q_init(self):
		self.Q_value = np.full((36,5), 0.0)
		pass


	def get_Q_value(self, state, action):
		q_val = self.Q_value[state.position][action]
		return q_val	


	def set_Q_value(self, state, action, val):
		self.Q_value[state.position][action] = copy.deepcopy(val)
	

	def agent_start(self, observation):
		#print "Agent start"
		curr_state = observation[0]
		self.accumulated_reward = observation[1]
		reward = observation[1]

		#e-greedy for selecting actions

		if random.random() > self.epsilon:
			temp = 0
			temp_max = -9999
			max_action = []
			i = 0
			for i in range(self.numActions):
				if temp_max < self.get_Q_value(curr_state, i):
					temp_max = self.get_Q_value(curr_state, i)

			for i in range(self.numActions):
				if self.get_Q_value(curr_state, i) == temp_max:
					max_action.append(i)

			temp = max_action[random.randrange(len(max_action))]
		else:
			temp = random.randrange(self.numActions)

		self.last_action = copy.deepcopy(temp)
		self.last_observation = copy.deepcopy(observation)
		self.numstep = 1

		self.track.append([curr_state.position, temp, reward])
		return temp

	

	def agent_step(self, observation, test):
		last_state = self.last_observation[0]
		curr_state = observation[0]
		reward = observation[1] # might be multiplied by some empathetic function...
		new_action = None
		#print reward

		self.accumulated_reward += reward

		if random.random() > self.epsilon:
			temp = 0
			temp_max = -9999
			max_action = []

			
			for i in range(self.numActions):
				if temp_max < self.get_Q_value(curr_state, i):
					temp_max = self.get_Q_value(curr_state, i)

			for i in range(self.numActions):
				if self.get_Q_value(curr_state, i) == temp_max:
					max_action.append(i)
			

			temp = max_action[random.randrange(len(max_action))]
		else:
			temp = random.randrange(self.numActions)

		new_action = temp

		#update Q value
		pre_Q = copy.deepcopy(self.get_Q_value(last_state, self.last_action))
		
		'''
		# Q learning
		max_Q = -9999.0
		for i in range(self.numActions):
			if max_Q <= self.get_Q_value(curr_state, i):
				max_Q = self.get_Q_value(curr_state, i)
		'''

		# expected sarsa
		max_Q = -9999.0
		max_a = 0
		exp_Q = 0.0

		for i in range(self.numActions):
			if max_Q <= self.get_Q_value(curr_state, i):
				max_Q = self.get_Q_value(curr_state, i)
				max_a = i

		for a in range(self.numActions):
			if a == max_a:
				exp_Q += (1 - self.epsilon + self.epsilon/self.numActions) * self.get_Q_value(curr_state, a)
			else:
				exp_Q += self.epsilon/self.numActions * self.get_Q_value(curr_state, a)

		new_Q = pre_Q + self.alpha * (reward + exp_Q - pre_Q)

		#Sarsa
		#new_Q = pre_Q + self.alpha * (reward + self.get_Q_value(curr_state, new_action) - pre_Q)


		if not test:
			self.set_Q_value(last_state, self.last_action, new_Q)
		
		self.last_action = copy.deepcopy(new_action)
		self.last_observation = copy.deepcopy(observation)

		self.track.append([curr_state.position, new_action, reward])
		self.numstep += 1

		#print new_action, new_Q, reward, new_behavior_value
		#print last_state.position, self.last_action, curr_state.position, new_action


		return new_action

	def agent_end(self, observation, test):
		#no update for Sarsa agent at the end of the episode
		#update Q value
		
		last_state = self.last_observation[0]
		curr_state = observation[0]
		reward = observation[1] 
		pre_Q = self.get_Q_value(last_state, self.last_action)
		new_Q = pre_Q + self.alpha * (reward - pre_Q)

		if not test:
			self.set_Q_value(last_state, self.last_action, new_Q)

		self.accumulated_reward += reward

		if test:
			with open("track.csv", "wb") as f:
				writer = csv.writer(f)
				writer.writerows(self.track)

		self.track = []

		return self.accumulated_reward
		pass