# mdp grid world .py

"""
Author Shivam Goel
WSUID# 11483916
"""


import time
import math
import numpy as np
import random
from random import randint




# MAP = [
#     "+---------+",
#     "|S        |",
#     "|         |",
#     "||||| ||| |",
#     "|  P|     |",
#     "|   |PP   |",
#     "|   |GP P |",
#     "|   |   P |",
#     "|  P|PP   |",
#     "|         |",
#     "|         |",
#     "+---------+"
# ]


class attributes():
	"""docstring for attributes"""
	def __init__(self):
		self.actions = [0, 1, 2, 3]
		self.reward = 0.0
		self.terminal = False
		self.wall = False


class Gridworld():

	def __init__(self):
		self.grid = {}
		# settting up the grid world
		for i in range (10):
			for j in range (10):
				# print i, j 
				
				self.grid[(i, j)] = attributes()
				
				# wall
				if (i == 2 and (j != 0 or j!= 5 or j!=9)):
					self.grid[(i, j)].wall = True
					self.grid[(i, j)].actions = [0.0]
				# wall	
				elif (j == 4 and (i!= 0 or i!= 1 or i!=8 or i!=9)):
					self.grid[(i, j)].wall = True
					self.grid[(i, j)].actions = [0.0]
				
				# top row
				elif (i == 0 and j != 0):
					self.grid[(i, j)].actions = [1, 2, 3]
				# leftmost row
				elif (j == 0 and i != 0):
					self.grid[(i, j)].actions = [0, 1, 3]
				# rightmost row
				elif (j == 9 and i !=0):
					self.grid[(i, j)].actions = [0, 1, 2]
				# bottom row	
				elif (i == 9 and j !=0):
					self.grid[(i, j)].actions = [0, 2, 3]
				
				# corners
				elif (i == 0 and j == 0):
					self.grid[(i, j)].actions = [2, 3]
				elif (i == 9 and j == 9):
					self.grid[(i, j)].actions = [1, 2]
				elif (i == 0 and j == 9):
					self.grid[(i, j)].actions = [2, 2]
				elif (i == 9 and j == 0):
					self.grid[(i, j)].actions = [1, 3]

				# Negative reward states
				elif (i == 4 and (j ==5 or j == 6 )):
					self.grid[(i, j)].reward = -1.0
				elif (i == 5 and (j == 6 or j ==8)):
					self.grid[(i, j)].reward = -1.0
				elif (i == 7 and (j == 3 or j == 4 or j == 6)):
					self.grid[(i, j)].reward = -1.0
				elif (i == 3 and j == 3):
					self.grid[(i, j)].reward = -1.0
				elif (i == 6 and j == 8):
					self.grid[(i, j)].reward = -1.0
				
				# Goal state
				elif (i == 5 and j == 5):
					self.grid[(i, j)].reward = 1.0


	def move(self, action):

		i, j = self.current_state
		i2, j2 = self.current_state
	
		if action not in self.grid[self.current_state].actions or self.grid[(i, j)].terminal == True or self.grid[(i, j)].wall == True:
			i2, j2 = i, j
			return (i2, j2)
		
		else:

			if action == 0: 
			# if grid[i][j].action() == 1:
				# move up
				i2 = i - 1

			elif action == 1:
			# elif grid[i][j].action() == 2:
				# move down
				i2 = i + 1
			elif action == 2:
			# elif grid[i][j].action() == 3:
				# move left
				j2 = j - 1
			elif action == 3:
				# move right
				j2 = j + 1
		# if self.grid[(i, j)].terminal == True:
		
			return (i2, j2)
	 

	def reset(self):
		self.current_state = (0, 0)
		return self.current_state[0] * 10 + self.current_state[1], self.grid[self.current_state].reward, self.grid[self.current_state].terminal,self.grid[self.current_state].actions

	def step(self, action):
		self.current_state = self.move(action)
		next_state = self.current_state[0] * 10 + self.current_state[1]
		return (next_state, self.grid[self.current_state].reward, self.grid[self.current_state].terminal, self.grid[self.current_state].actions)
 

class QLearning():


	def __init__(self, env):

		self.qvalue = np.zeros((100, 4))
		self.alpha = 0.01
		self.beta = 0.9
		self.epsilon = 0.1 
								#0.2, 0.3
		self.T = 10
		self.env = env

	def qlearning(self):

		s, reward, terminal, actions = self.env.reset()
		# print actions

		while True:
			action = self.epsilongreedy(s, actions)
			# print action
			nextstate, reward, terminal, _ = self.env.step(action)
			# qvalue update
			
			self.qvalue[s][action] += self.alpha*(reward + self.beta * np.max(self.qvalue[nextstate]) - self.qvalue[s][action])
			if terminal:
				print self.qvalue
				s = self.env.reset()
		
			s = nextstate

	def epsilongreedy(self, s, actions):

	    random_num = np.random.random()
	    # print random_num
	    if random_num < self.epsilon:

	        return random.choice(actions)
	    else:
	        return np.argmax(self.qvalue[s])

	def boltzman_exploration(self, s):
		prob_of_action = []
		for i in range(4):
			prob_of_action_dash = (self.qvalue[s][i]/self.T)/ np.sum(self.qvalue[s])/self.T
			prob_of_action.append(prob_of_action_dash)
		return np.argmax(prob_of_action_dash)

def main():
	env = Gridworld()
	ql = QLearning(env)
	ql.qlearning()


if __name__ == '__main__':
	main()


'''
MAP = [
    "+---------+",
    "|S        |",
    "|         |",
    "||||| ||| |",
    "|  P|     |",
    "|   |PP   |",
    "|   |GP P |",
    "|   |   P |",
    "|  P|PP   |",
    "|         |",
    "|         |",
    "+---------+"
]

'''









