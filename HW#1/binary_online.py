import numpy as np
import math
import matplotlib as plt
import re
from numpy import array
import random
import time
import matplotlib.pyplot as plt

class BinaryClassifier():

	def __init__(self, path1, path2):
		#inititalization of the parameters
		self.w = np.zeros(128)
		self.t = max_iter = 20
		self.u = np.zeros(128)
		self.w2_aver = np.zeros(128)
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
				#g = list(list1[0])
				#extract the label in each x
				letter = g[2]
				#extract the x for each input data
				word = list(g[1][2:])
				word = map(float, word)
				word.append(letter)
				list_word.append(word)
				list_letter.append(letter)
				list_main.append(g)

		return list_letter, list_word, list_main


	def classifier_train(self, passive):

		vowels = ['a','e','i','o','u']
		#main function for classification
		# n = 100
		for i in range (self.t):

			mistake = 0

			# after each iteration do a random shuffle of the array
			random.shuffle(self.train_list_word)

			for j in range (len(self.train_list_word)):
				
				yhat_t = np.dot(self.w,self.train_list_word[j][0:128])

				if (any(n in self.train_list_word[j] for n in vowels)):
					y_t = 1 
				else:
					y_t = -1
				# condition of passive	
				if passive is 1:
					# condition for incorrect classification
					if (y_t * yhat_t <= 1):
						
						# found mistake in classification training
						mistake += 1

						self.tau = (1 - y_t * yhat_t) / math.pow(np.linalg.norm(self.train_list_word[j][0:128]),2)

						for k in range(len(self.w)):
							self.w[k] = self.w[k] + self.tau * self.train_list_word[j][k] * y_t
				# Standard perceptron
				else:
					if y_t * yhat_t <= 0:
						# mistake found in classification training
						mistake += 1
						for k in range (len(self.w)):
							self.w[k] = self.w[k] + self.tau * self.train_list_word[j][k] * y_t
			# n += 100
	
			# appended the total mistakes for ith iteration of the training classification 			
			self.mistakes_train.append(mistake)
			# for mistakes calculation on testing data
			mistake_test = self.classifier_test()
			# update the list with mistakes for each iteration
			self.mistakes_test.append(mistake_test)

		return self.w

	def classifier_train_average_perceptron(self):
		vowels = ['a','e','i','o','u']
		c = 1
		for i in range (self.t):
			
			mistake = 0

			# after each iteration do a random shuffle of the array
			random.shuffle(self.train_list_word)

			for j in range (len(self.train_list_word)):

				if (any(n in self.train_list_word[j] for n in vowels)):
					y_t = 1 
				else:
					y_t = -1
				
				yhat_t = np.dot(self.w,self.train_list_word[j][0:128])

				if y_t * yhat_t <= 0:
					mistake += 1
					# update the weight vector
					for k in range (len(self.w)):
						self.w[k] += self.train_list_word[j][k] * y_t
					# update the u vector
					for k in range (len(self.u)):
						self.u[k] += y_t * c * self.train_list_word[j][k]
				# update c
				c += 1
			# print mistake
			self.mistakes_train.append(mistake)
			for k in range (len(self.w2_aver)):
				self.w2_aver[k] = self.w[k] - (self.u[k] / c)
			
			mistake_test = self.classifier_test()
			# print "        {}".format(mistake_test)
			self.mistakes_test.append(mistake_test)
			# print ("       {}".format(mistake_test))	
		
		return self.w



	def classifier_test(self):
		vowels = ['a','e','i','o','u']
		mistake = 0
		# basically you have to calculate the mistakes in the classification, technically the number of wrong classifications done by each algorithm.
		# so using the weights, we will do the predictions,
		for j in range(len(self.test_list_word)):
			yhat_t = np.dot(self.w2_aver, self.test_list_word[j][0:128])
			# print yhat_t
			# check for mistake
			if (any(n in self.test_list_word[j] for n in vowels)):
				y_t = 1 
			else:
				y_t = -1
			if (yhat_t * y_t) <= 0:
				mistake += 1
		return mistake


def main():



	average_mistakes_test = []
	average_mistakes_train = []

	p = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

	path1 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_train.txt"
	path2 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_test.txt"
	bc = BinaryClassifier(path1, path2)
	


	for i in range(len(p)):
		path1 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold{}_sm_train.txt".format(p[i])
		path2 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold{}_sm_test.txt".format(p[i])
		bc = BinaryClassifier(path1, path2)
		# bc.classifier_train_average_perceptron()
		# 1 for PA and 0 for Perceptron
		# bc.classifier_train(0)
		bc.classifier_train_average_perceptron()
		# print (bc.w)
		# print (bc.mistakes)
		average_mistakes_train.append(bc.mistakes_train)
		average_mistakes_test.append(bc.mistakes_test)
	

	average_mistakes_train = np.array(np.mean(average_mistakes_train, axis = 0), dtype = float)
	average_mistakes_train = np.array(np.subtract(len(bc.train_list_word), average_mistakes_train), dtype = float)
	average_mistakes_train = np.array(np.divide(average_mistakes_train, len(bc.train_list_word)), dtype = float)
	plt.plot(average_mistakes_train)

	average_mistakes_test = np.array(np.mean(average_mistakes_test, axis = 0), dtype = float)
	average_mistakes_test = np.array(np.subtract(len(bc.test_list_word), average_mistakes_test), dtype = float)
	average_mistakes_test = np.array(np.divide(average_mistakes_test, len(bc.test_list_word)), dtype = float)	
	plt.plot(average_mistakes_test)

	



	# average_mistakes_test = np.array(50, dtype = 'f')
	# average_mistakes_train = []

	# path1 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_train.txt"
	# path2 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_test.txt"
	# bc = BinaryClassifier(path1, path2)
	# bc.classifier_train(0)
	# average_mistakes_test = np.array((np.subtract(len(bc.test_list_word), bc.mistakes_test)), dtype = float)
	# average_mistakes_test = (average_mistakes_test / len(bc.test_list_word))
	# plt.plot(average_mistakes_test)



	# average_mistakes_test = np.array(50, dtype = 'f')
	# average_mistakes_train = []

	# path1 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_train.txt"
	# path2 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_test.txt"
	# bc = BinaryClassifier(path1, path2)
	# bc.classifier_train(1)
	# average_mistakes_test = np.array(np.subtract(len(bc.test_list_word), bc.mistakes_test), dtype = float)
	# average_mistakes_test = (average_mistakes_test / len(bc.test_list_word))
	# plt.plot(average_mistakes_test)


	plt.xlabel("Number of iterations", fontsize = 15)
	plt.ylabel(" Accuracy", fontsize = 15)
	plt.title("Averaged Perceptron (Training vs Testing)", fontsize = 25)
	plt.grid(True)
	plt.legend(['A.Perceptron (Training)', 'A. Perceptron (Testing)'])
	plt.show()  

	#TODO:make a plotting function here



if __name__ == '__main__':
	main()





		
