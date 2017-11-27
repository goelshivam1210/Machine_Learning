'''
https://stackoverflow.com/questions/2621674/how-to-extract-elements-from-a-list-in-python



'''

import numpy as np
import math
import re
import math
import sys


list_word = []
list_letter = []
list_main = []
list_efficient = np.zeros(shape = (127,1), dtype = float)
f = open("/home/goelshivam12/Desktop/ML_Homework/HW#1/ocr_fold1_sm_train.txt")		#code for parsing

for line in f:
	if  line.strip():
		list1 = line.split(" ")
		g = list(list1[0])
		# print (g)
		letter = g[133:134]
		word = g[5:132]
		list_word.append(word)
		list_letter.append(letter)
		list_main.append(g)
	

# print(list_word[3])
# print (len(list_word[3]))
# print (list_word[4])
# print (list_letter[3])
# print (list_letter[4])
# print (len(list_word), len(list_letter))
print (map(int, list_word[1]) * list_efficient)