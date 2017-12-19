#import Tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import sys, os


def save_result(data,path):
	f = open(path,"w")
	for i in range(len(data)):
		f.write(str(data[i])+'\n')
	f.close()
	pass

def save_csv(data,path):
	with open(path, "wb") as f:
		writer = csv.writer(f)
		writer.writerows(data)

def manhattan_distance(start, end):
    sx, sy = start
    ex, ey = end
    return abs(ex - sx) + abs(ey - sy)
		

def to_cord(position, width):
	#(0,0) lays on the southwest corner
	x = position % width
	y = position // width
	#print position,"to",x,y
	return x, y

def to_pos(x, y, width):
	# starts from 0
	position = y * width + x
	return position

def visualize_result(data, x_label, y_label):
	plt.plot(data)
	plt.ylabel(x_label)#'# of steps'
	plt.xlabel(y_label)#'# of episodes'
	plt.show()
	pass
