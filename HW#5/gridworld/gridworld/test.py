import numpy as np
import utils
import csv
import copy
import sys
from time import sleep
import experiment

width = 6

def board_init(width, height):
	board = [[' ']* width for i in range(height)]

	obstacles = [13, 21, 35]
	for i in obstacles:
		x,y = utils.to_cord(i, 6)
		board[5-y][x] = 'X'

	pits = [3,25]
	for i in pits:
		x,y = utils.to_cord(i, 6)
		board[5-y][x]  = '#'

	food = [10]#,20]
	for i in food:
		x,y = utils.to_cord(i, 6)
		board[5-y][x] = 'F'

	charger = [33]
	for i in charger:
		x,y = utils.to_cord(i, 6)
		board[5-y][x]  = 'G'


	return board

def board_print(board):
	global width
	line = ['--. ' * width]
	print ''.join(line)
	for item in board:
		print ' | '.join(map(str, item))
		print ''.join(line)

def action(num):
	if num == '0':
		a = 'Stay'
	elif num == '1':
		a = 'Right'
	elif num == '2':
		a = 'Down'
	elif num == '3':
		a = 'Left'
	elif num == '4':
		a = 'Up'

	return a


if __name__ == '__main__':
	board = board_init(6, 6)


	if sys.argv[1] != "0":
		file_name = sys.argv[1]
	else:
		env_parameter = [6,6,0,33]
		agent_parameter = [5, 0.2, 1.0, 0.9, 0.0] #numActions, epsilon, gamma, alpha, init value

		task_env, task_agent = experiment.RL_init(env_parameter, agent_parameter)
		task_agent.behavior_value = np.load("B_value.npy")
		task_agent.Q_value = np.load("Q.npy")
		experiment.test(task_env, task_agent, 200)
		file_name  = "track.csv"

	with open(file_name, 'rb') as f:
		reader = csv.reader(f)
		track = list(reader)


	print "Start..."

	i = 0
	count = 0
	for item in track:
		current_board = copy.deepcopy(board)
		i += 1
		x, y = utils.to_cord(int(item[0]), 6)
		current_board[5-y][x]  = '@'
		a = action(item[4])
		board_print(current_board)
		print "Step", i, ": ",a, "Battery level: ", float(item[3]), "Reward: ", float(item[5]),"Dir_food: ", float(item[1])
		if int(item[0]) == 10:
			count += 1
		sleep(0.1)
		print "Next"

		#raw_input("Next")
	print count



