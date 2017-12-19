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
import matplotlib.pyplot as plt




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
		self.reward = -0.04
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
				# print ((i==2) and ((j==1) or (j==2) or (j==3) or (j==4) or (j==6) or (j==7) or (j==8) )) 
				if ((i==2) and ((j==1) or (j==2) or (j==3) or (j==4) or (j==6) or (j==7) or (j==8) )):
					self.grid[(i, j)].wall = True
					self.grid[(i, j)].actions = [0]
			# wall	
				elif ((j==4) and ((i==2) or (i==3) or (i==4) or (i==5) or (i==6) or (i==7))):
					self.grid[(i, j)].wall = True
					self.grid[(i, j)].actions = [0]
				
			# Negative reward states
				elif ((i == 4) and ((j ==5) or (j == 6))):
					self.grid[(i, j)].reward = -1.0
				elif ((i == 5) and ((j == 6) or (j ==8))):
					self.grid[(i, j)].reward = -1.0
				elif ((i == 7) and ((j == 3) or (j == 4) or (j == 6))):
					self.grid[(i, j)].reward = -1.0
				elif (i == 3 and j == 3):
					self.grid[(i, j)].reward = -1.0
				elif (i == 6 and j == 8):
					self.grid[(i, j)].reward = -1.0
				
				# Goal state
				elif (i == 5 and j == 5):
					self.grid[(i, j)].reward = 1.0
					self.grid[(i, j)].terminal = True
				else:
					self.grid[(i, j)].actions = [0, 1, 2, 3]


	def move(self, action):

		i, j = self.current_state
		i2, j2 = 0, 0

# if the current state is in the corners, for corners, and if the actions are accordingly then do nothing.
		if j==0 and i==0:
			if action == 0:
				i2 = i
				j2 = j
			elif action == 1:
				i2 = i+1
				j2 = j
			elif action ==2:
				i2 = i
				j2 = j
			elif action == 3:
				i2 = i
				j2 = j+1

		elif j==9 and i==9:
			if action == 0:
				i2 = i-1
				j2 = j
			elif action == 1:
				i2 = i
				j2 = j
			elif action ==2:
				i2 = i
				j2 = j-1
			elif action == 3:
				i2 = i
				j2 = j

		elif j==0 and i==9:
			if action == 0:
				i2 = i-1
				j2 = j
			elif action == 1:
				i2 = i
				j2 = j
			elif action ==2:
				i2 = i
				j2 = j
			elif action == 3:
				i2 = i
				j2 = j+1

		elif j==9 and i==0:
			if action == 0:
				i2 = i
				j2 = j
			elif action == 1:
				i2 = i +1
				j2 = j
			elif action ==2:
				i2 = i
				j2 = j-1
			elif action == 3:
				i2 = i
				j2 = j

		
		# top row, only down, left, right moves
		elif i==0 and j!=0:
			if action == 0:
				i2 = i
				j2 = j
			elif action == 1:
				i2 = i+1
				j2 = j
			elif action ==2:
				i2 = i
				j2 = j-1
			elif action == 3:
				i2 = i
				j2 = j+1

		# leftmost row		
		elif j==0 and i!=0:
			if action == 0:
				i2 = i-1
				j2 = j
			elif action == 1:
				i2 = i+1
				j2 = j
			elif action ==2:
				i2 = i
				j2 = j
			elif action == 3:
				i2 = i
				j2 = j+1
		# rightmost row
		elif j==9 and i!=0:
			if action == 0:
				i2 = i-1
				j2 = j
			elif action == 1:
				i2 = i+1
				j2 = j
			elif action ==2:
				i2 = i
				j2 = j-1
			elif action == 3:
				i2 = i
				j2 = j
		# bottom row, up left right
		elif j!=0 and i==9:
			if action == 0:
				i2 = i-1
				j2 = j
			elif action == 1:
				i2 = i
				j2 = j
			elif action ==2:
				i2 = i
				j2 = j-1
			elif action == 3:
				i2 = i
				j2 = j+1
		else:
			if action == 0:
				i2 = i-1
				j2 = j
			elif action == 1:
				i2 = i+1
				j2 = j
			elif action ==2:
				i2 = i
				j2 = j-1
			elif action == 3:
				i2 = i
				j2 = j+1

		if self.grid[(i2, j2)].wall == True:
			return i, j
		else:
			return i2, j2



	def reset(self):
		self.current_state = (0, 0)
		return self.current_state[0] * 10 + self.current_state[1], self.grid[self.current_state].reward, self.grid[self.current_state].terminal,self.grid[self.current_state].actions

	def step(self, action):
		# print action
		# print self.grid[self.current_state].actions
		self.current_state = self.move(action)

		# print self.current_state
		next_state = self.current_state[0] * 10 + self.current_state[1]
		# print next_state
		return (next_state, self.grid[self.current_state].reward, self.grid[self.current_state].terminal, self.grid[self.current_state].actions)
 

# class Test():


class QLearning():


	def __init__(self, env):

		self.qvalue = np.full((100, 4), 0.01)

		self.alpha = 0.01
		self.beta = 0.9
		self.epsilon = 0.1 
								#0.2, 0.3
		self.T = 10
		self.env = env

	def qlearning(self):

		s, reward, terminal, actions = self.env.reset()
		# print actions
		total_rew = []
		rew = 0.
		n = 0

		while True:
			# either select action using boltzmanns or select using epsilon greedy
			action = self.epsilongreedy(s, self.env.current_state)
			nextstate, reward, terminal, _ = self.env.step(action)
			rew += reward
			# qvalue update
			pi = [2, 3, 4, 5, 6, 7]
			pj = [1, 2, 3, 4, 6, 7, 8]
			# if nextstate/10 in pi  and nextstate%10 in pj:
			# print nextstate/10, nextstate%10
			# print action
			# time.sleep(0.1)

			
			self.qvalue[s][action] += self.alpha*(reward + self.beta * np.max(self.qvalue[nextstate]) - self.qvalue[s][action])
			if terminal:
				total_rew.append(rew)
				rew = 0
				n += 1
				print n, rew
				# print self.qvalue
				s = self.env.reset()
		
			s = nextstate

			if n > 100000000:
				plt.plot(total_rew)
				print self.qvalue
				plt.show()
				break
			# print self.qvalue

	def epsilongreedy(self, s, current_state):

	    random_num = np.random.random()
	    # print self.env.grid[(0, 1)].actions
  	    actions = self.env.grid[current_state].actions

    
    # print random_num
	    if random_num < self.epsilon:

	        return random.choice(actions)
	    else:
	    	# if self.qvalue[s] 
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









