import numpy as np
import math
import scipy
import matplotlib as plt
import re
from numpy import array
import random
import time
import matplotlib.pyplot as plt
import pandas

class MultiClassifier():

	def __init__(self, path1, path2):
		#inititalization of the parameters
		# inititalize the weight vector with the dimension of 26*128
		self.w = np.zeros(shape = (26, 128), dtype = float)
		self.u = np.zeros(shape = (26, 128), dtype = float)
		self.w2_aver = np.zeros(shape = (26, 128), dtype = float)


		self.t = max_iter = 10

		#learning rate
		self.tau = 1

		self.train_list_letter, self.train_list_word, _ = self.parse(path1)
		self.test_list_letter, self.test_list_word, _ = self.parse(path2)

		self.mistakes_test = []
		self.mistakes_train = []

	def parse(self, file):
		# list of the x's
		list_word = []
		# list of the y's
		list_letter = []
		#holds the whole list
		list_main = []

		f = open(file)		#code for parsing

		for line in f:
			# to remove the blank lines
			if line.strip():
				g = line.split("\t")
				#convert each element in the list to a list
				#extract the label in each x
				letter = g[2]
				# map the letter to a number 
				# a -> 1, b -> 2, c -> 3, ...
				letter = ord(letter) - 97
				#extract the x for each input data
				word = list(g[1][2:])
				word = np.array(map(float, word))

				list_word.append(word)
				list_letter.append(letter)
				list_main.append(g)

		return list_letter, list_word, list_main


	def classifier_train(self, passive):

		# import sys
		# sys.exit()
		#main function for classification
		number_iter = 100
		for i in range (self.t):
			mistake = 0

			# after each iteration do a random shuffle of the array
			from sklearn.utils import shuffle
			self.train_list_word, self.train_list_letter = shuffle (self.train_list_word, self.train_list_letter, random_state = 0)

			for j in range(number_iter):
			# for j in range (len(self.train_list_word)):

				yhats = []

				for k in range (len (self.w)):

					temp_wt = np.dot(self.w[k], self.train_list_word[j])
					yhats.append(temp_wt)

				yhat_t = np.argmax(yhats)

				# if (yhat_t != self.train_list_letter[j]):
					
				# 	mistake += 1
					# checking the condition 
				if passive is 1:

					# condition check

				
					if ((np.dot(self.w[self.train_list_letter[j]], self.train_list_word[j]) - np.dot(self.w[yhat_t], self.train_list_word[j])) < 1) :
						mistake += 1

						if (yhat_t != self.train_list_letter[j]):


														
							f_y = np.zeros((26, 128))
							f_yhat = np.zeros((26, 128))
							f_y[self.train_list_letter[j]] += self.train_list_word[j]
							f_yhat[yhat_t] += self.train_list_word[j]	
							# print len(f_y)
							# print len(self.w)					

							tau_temp = np.dot(self.w[self.train_list_letter[j]], self.train_list_word[j]) - np.dot(self.w[yhat_t], self.train_list_word[j])
							# tau_temp = np.dot(self.w, f_y) - np.dot(self.w, f_yhat)
					
							# computing norm
							# f_y = np.zeros((26, 128))
							# f_yhat = np.zeros((26, 128))
							# f_y[self.train_list_letter[j]] += self.train_list_word[j]
							# f_yhat[yhat_t] += self.train_list_word[j]
							n = f_y - f_yhat

							n = math.pow(np.linalg.norm(n), 2)
							# print n
						
							self.tau = (1 - tau_temp) / n

							# update the weights
							# self.w[self.train_list_letter[j]] = self.w[self.train_list_letter[j]] + self.tau * self.train_list_word[j]
							# self.w[yhat_t] -= self.tau * self.train_list_word[j]

							self.w +=  self.tau * (f_y - f_yhat)

				# case where passive is not true (Standard perceptron)		
				else:
					if (yhat_t != self.train_list_letter[j]):

						mistake += 1

					# adding to the correct index
						self.w[self.train_list_letter[j]]+= self.tau * self.train_list_word[j]
						# subtracting from the wrong index 
						self.w[yhat_t] -= self.tau * self.train_list_word[j]
			number_iter += 100

			self.mistakes_train.append(mistake)

			mistake_test = self.classifier_test()
			# print "        {}".format(mistake_test)
			self.mistakes_test.append(mistake_test)
		return self.w


	def classifier_test(self):
		mistake = 0
		for j in range (len(self.test_list_word)):
			yhats = []
			for k in range (len (self.w)):
				temp_wt = np.dot(self.w[k], self.test_list_word[j])
				yhats.append(temp_wt)
			yhat_t = np.argmax(yhats)
				# print yhat_t
				# print self.train_list_word[j][128]

				# checking the condition 
			if (yhat_t != self.test_list_letter[j]):
				mistake += 1

		return mistake

	def classifier_train_average_perceptron(self):
		c = 1
		for i in range (self.t):
			mistake = 0
			from sklearn.utils import shuffle
			self.train_list_word, self.train_list_letter = shuffle (self.train_list_word, self.train_list_letter, random_state = 0)			
			# random.shuffle(self.train_list_word)

			for j in range (len(self.train_list_word)):
				yhats = []

				for k in range (len (self.w)):

					temp_wt = np.dot(self.w[k], self.train_list_word[j])
					yhats.append(temp_wt)

				yhat_t = np.argmax(yhats)



				# yhat_t = self.train_list_letter[j] * (np.dot(self.w[j], self.train_list_word[j]))
				# condition check
				if (yhat_t != self.train_list_letter[j]):

					mistake += 1
			
					self.w[int(yhat_t)] -= self.train_list_word[j]
					self.w[self.train_list_letter[j]] += self.train_list_word[j]
					
					self.u[int(yhat_t)] -= c * self.train_list_word[j]
					self.u[self.train_list_letter[j]] += c * self.train_list_word[j] 
				c += 1
			self.mistakes_train.append(mistake)
			self.w2_aver = self.w - (self.u / c)

			mistake_test = self.classifier_test()
			self.mistakes_test.append(mistake_test)

		return self.w



def main():


	average_mistakes_test = np.array(50, dtype = 'f')

	path1 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_train.txt"
	path2 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_test.txt"
	mc = MultiClassifier(path1, path2)
	mc.classifier_train(0)
	average_mistakes_test = np.array((np.subtract(len(mc.test_list_word), mc.mistakes_test)), dtype = float)
	average_mistakes_test = (average_mistakes_test / len(mc.test_list_word))
	plt.plot(average_mistakes_test)



	average_mistakes_test = np.array(50, dtype = 'f')

	path1 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_train.txt"
	path2 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_test.txt"
	mc =  MultiClassifier(path1, path2)
	mc.classifier_train(1)
	average_mistakes_test = np.array(np.subtract(len(mc.test_list_word), mc.mistakes_test), dtype = float)
	average_mistakes_test = (average_mistakes_test / len(mc.test_list_word))
	plt.plot(average_mistakes_test)








	# average_mistakes_test = []
	# average_mistakes_train = []

	# p = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

	# for i in range(len(p)):
	# 	path1 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold{}_sm_train.txt".format(p[i])
	# 	path2 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold{}_sm_test.txt".format(p[i])
	# 	mc = MultiClassifier(path1, path2)
	# 	# 0 for Perceptron, 1 for PA
	# 	# mc.classifier_train_average_perceptron()
	# 	mc.classifier_train(0)
	# 	# print (bc.w)
	# 	# print (bc.mistakes)

	# 	average_mistakes_train.append(mc.mistakes_train)
	# 	average_mistakes_test.append(mc.mistakes_test)

	# average_mistakes_train = np.array(np.mean(average_mistakes_train, axis = 0), dtype = float)
	# average_mistakes_train = np.array(np.subtract(len(mc.train_list_word) ,average_mistakes_train), dtype = float)
	# average_mistakes_train = np.array(np.divide(average_mistakes_train, len(mc.train_list_word)), dtype = float)
	

	# average_mistakes_test = np.array(np.mean(average_mistakes_test, axis = 0), dtype = float)
	# average_mistakes_test = np.array(np.subtract(len(mc.test_list_word), average_mistakes_test), dtype = float)
	# average_mistakes_test = np.array(np.divide(average_mistakes_test, len(mc.test_list_word)), dtype = float)
	# # print average_mistakes_test

	# plt.plot(average_mistakes_train)
	# plt.plot(average_mistakes_test)

	plt.xlabel("Number of Iterations", fontsize = 15)
	plt.ylabel(" Accuracy", fontsize = 15)
	plt.title("General Learning Curve (PA vs P)", fontsize = 25)
	plt.ylim([0.1, 0.8])
	plt.grid(True)
	plt.legend(['Std. Perceptron', ' PA'])
	plt.show()  



	#make a plotting function here



if __name__ == '__main__':
	main()
