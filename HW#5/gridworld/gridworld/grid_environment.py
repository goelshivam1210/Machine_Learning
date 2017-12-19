#import Tkinter as tk
import numpy as np
import math
import random
import copy
import utils


class States:
	def __init__(self, position):
		self.position = position # current position

class Gridworld_Env:
	def __init__(self, width, height, start_pos, goal_pos):
		#print "Gridworld environment init..."
		self.width = width
		self.height = height
		self.space_size = self.width * self.height
		self.objects = [None for i in range(self.space_size)]
		self.init_state = None
		self.term_state = None
		self.stepcounter = 0

		if start_pos == "rd":
			self.start_pos = random.randrange(self.space_size)
		else:
			self.start_pos = start_pos

		if goal_pos == "rd":
			self.goal_pos = random.randrange(self.space_size)
			while self.goal_pos == self.start_pos:
				self.goal_pos = random.randrange(self.space_size)
		else:
			self.goal_pos = goal_pos

		#treat start and goal as types of objects, with value 0
		#self.objects[self.start_pos] = ["start pos", 0]
		#self.objects[self.goal_pos] = ["goal pos", 0]

		#print "Grids",self.width,"*",self.height
		#print "The start position is ",self.to_cord(self.start_pos),"\nThe goal position is ",self.to_cord(self.goal_pos)

	def add_obj(self, pos, value, obj_type):
		#obj_type = {"obstacle", "pit", "food", "start_pos", "goal_pos"}
		self.objects[pos] = [obj_type, value]
		
	def to_cord(self, position):
		#(0,0) lays on the southwest corner
		x = position % self.width
		y = position // self.width
		#print position,"to",x,y
		return x, y

	def to_pos(self, x, y):
		# starts from 0
		position = y * self.width + x
		return position

	def get_numStates(self):
		numStates = self.space_size
		return numStates

	'''
	Here starts the RL environment part
	'''
	def moving(self, s, a):
		curr_pos = copy.deepcopy(s.position)
		curr_x, curr_y = self.to_cord(curr_pos)
		next_x = 0
		next_y = 0
		reward = 0

		#or use dictionary for switching actions
		if a == 0:
			#stay
			next_x = curr_x
			next_y = curr_y
		elif a == 1:
			#right
			next_x = curr_x + 1
			next_y = curr_y
		elif a == 2:
			#down
			next_x = curr_x 
			next_y = curr_y - 1
		elif a == 3:
			#left
			next_x = curr_x - 1
			next_y = curr_y
		elif a == 4:
			#up
			next_x = curr_x 
			next_y = curr_y + 1

		#check bound
		if (next_x >= self.width or next_x < 0) or (next_y >= self.height or next_y < 0):
			#print "bump wall!"
			temp_pos = curr_pos
		else:
			temp_pos = self.to_pos(next_x,next_y)


		# initial a new state
		new_state = States(0)#position, dir_food, charging, battery

		# state value change
		
		
		#check objects
		if not self.objects[temp_pos]:
			new_state.position = temp_pos
			return new_state, reward

		elif self.objects[temp_pos][0] == "obstacle":
			#print "bump obstacles!"
			temp_pos = curr_pos

		elif self.objects[temp_pos][0] == "pit":
			#print "fall in pit!"
			reward = self.objects[temp_pos][1]

		elif temp_pos == self.goal_pos:
			reward = 5

		new_state.position = temp_pos

		return new_state, reward

	def env_start(self):
		self.init_state = States(self.start_pos) 
		self.stepcounter = 0

		return self.init_state, 0

	def env_step(self, state, action):
		new_state, reward = self.moving(state, action)
		if self.terminal_check(new_state):
			reward += 0

		#print state.position, action, new_state.position, reward
		self.stepcounter += 1
		return new_state, reward, self.terminal_check(new_state)

	def env_end(self, state):
		self.term_state = self.init_state

		pass

	def terminal_check(self, state):
		if state.position == self.goal_pos:
			return True
		else:
			return False
		
		pass
