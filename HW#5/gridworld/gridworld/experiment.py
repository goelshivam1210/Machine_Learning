import grid_environment as env
import sarsa_agent as agent
#import grid_ui
import utils
import random
import time
import matplotlib.pyplot as plt
#from tempfile import TemporaryFile
import numpy as np

numSteps = 0
accumulated_reward = 0

def RL_init(env_parameter, agent_parameter):
	#init the gridworld environment
	task_env = env.Gridworld_Env(env_parameter[0],env_parameter[1],env_parameter[2],env_parameter[3])
	#obj_type = {"obstacle", "consumable_resource", "permanent_resource", "pit","start_pos", "goal_pos"}
	#task_env.random_objects(2, "obstacle", 3)
	task_env.add_obj(13, 0, "obstacle")
	task_env.add_obj(21, 0, "obstacle")
	task_env.add_obj(35, 0, "obstacle")
	task_env.add_obj(3, -1, "pit")	# -1 reward for pits
	task_env.add_obj(25, -1, "pit")

	numStates = task_env.get_numStates()
	numActions = agent_parameter[0]
	epsilon = agent_parameter[1]
	gamma = agent_parameter[2]
	alpha = agent_parameter[3]
	init_value = agent_parameter[4]

	task_agent = agent.Sarsa_Agent(numStates, numActions, epsilon, gamma, alpha, init_value)

	return task_env, task_agent

def RL_epispdes(task_env, task_agent, max_steps):
	global numSteps, accumulated_reward
	numSteps = 0
	accumulated_reward = 0
	observation = task_env.env_start()
	#print "Environment, start state",observation[0],"start reward",observation[1]
	action = task_agent.agent_start(observation)
	#print "Agent, start action", action 

	terminal = False
	test = False

	while not terminal and numSteps <= max_steps:
		numSteps += 1 
		observation = task_env.env_step(observation[0], action)
		action = task_agent.agent_step(observation, test)
		terminal = task_env.terminal_check(observation[0])
	else:
		task_env.env_end(observation[0])
		accumulated_reward = task_agent.agent_end(observation, test)
		#print task_env.foodcounter

def test(env, agent, max_steps):
	global numSteps, accumulated_reward
	numSteps = 0
	accumulated_reward = 0
	observation = env.env_start()
	action = agent.agent_start(observation)
	#agent.epsilon = 0.0

	terminal = False
	test = True

	while not terminal:
		numSteps += 1 
		observation = env.env_step(observation[0], action)
		action = agent.agent_step(observation, test)
		terminal = observation[2]
		if terminal or numSteps >= max_steps:
			print "test end"
			break
	
	env.env_end(observation[0])
	accumulated_reward = agent.agent_end(observation, test)
	print numSteps

	pass
	
		


def main_experiment(env_parameter, agent_parameter, experiment_parameter):
	print "EXPERIMENT START"
	numRuns = experiment_parameter[0]
	numEpisodes = experiment_parameter[1]
	max_steps = experiment_parameter[2]

	average_steps = [0.0 for i in range(numEpisodes)]
	average_acc_reward = [0.0 for i in range(numEpisodes)]
	global numSteps, accumulated_reward

	for i in range(numRuns):
		print "Runs",i,"start..."
		task_env, task_agent = RL_init(env_parameter, agent_parameter)
		for j in range(numEpisodes):
			RL_epispdes(task_env, task_agent, max_steps)
			average_steps[j] += numSteps/(numRuns+1.0)
			average_acc_reward[j] += accumulated_reward/(numRuns+1.0)
		if i == numRuns - 1:
			print "test mode"
			test(task_env, task_agent, max_steps)

	utils.save_result(average_steps,"avg_steps.txt")
	utils.save_result(average_acc_reward,"avg_acc_reward.txt")
	np.save("Q.npy", task_agent.Q_value)

	plot_steps = plt.figure(1)
	plt.plot(average_steps)
	plt.ylabel('Steps')#'# of steps'
	plt.xlabel('Episodes')#'# of episodes'
	#plot_steps.show()

	plot_reward = plt.figure(2)
	plt.plot(average_acc_reward)
	plt.ylabel('Accumulated Rewards')#'# of steps'
	plt.xlabel('Episodes')#'# of episodes'
	#plot_reward.show()

	plt.show()


	print "EXPERIMENT END"

	pass

if __name__ == '__main__':
	random.seed(time.time())
	env_parameter = [6,6,0,33]
	agent_parameter = [5, 0.2, 1.0, 0.9, 0.0] #numActions, epsilon, gamma, alpha, init value
	experiment_parameter = [2,200,200] #numRuns, numEpisodes, maxSteps
	main_experiment(env_parameter, agent_parameter,experiment_parameter)