from svmutil import *
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time

class KernelizedPerceptron_multi():

	def __init__(self, path1, path2):

		self.train_list_letter, self.train_list_word = self.parse(path1)
		self.test_list_letter, self.test_list_word = self.parse(path2)


		self.alpha = np.zeros(shape = (26, len(self.train_list_word)), dtype = float)
		self.maxiter = 20

		self.mistakes_train = []
		self.mistakes_test = [] 

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
				word = (map(float, word))
				# print word
				# if validation set is being computed
				list_word.append(word)
				list_letter.append(letter)
				list_main.append(g)
		
		list_letter = np.array(list_letter, dtype = int)
		list_word = np.array(list_word, dtype = int)
		# list_letter = np.array(list_letter, dtype = float)				
		return list_letter, list_word



	def classifier_train (self):

		for iter in range (self.maxiter):
			
			mistake = 0

			for i in range (len(self.train_list_word)):
				# now we have to compute the activation function
				# computation of activation function
				A = []
				a = 0
				if (len(self.alpha)) > 26:
					print len(self.alpha)
				for j in range (len (self.alpha)):

					# computation of the activation for each class label
					for k in range (len (self.alpha[j])):

						if self.alpha[j][k] != 0:
							
							a_ = np.dot(self.train_list_word[k], self.train_list_word[i]) + 1

							a__ = math.pow(a_, 2)

							a += self.alpha[j][k] * a__ 

					A.append(a)

				# select the maximum value for predicting the class label
				activation = np.argmax(A)
				if  activation != self.train_list_letter[i]:
					mistake += 1
					# then its a mistake hence, we need to punish the classifier
					# add the value of 1 to the nth index of the alphas
					self.alpha[self.train_list_letter[i]][i] += 1  
					# subtract the value of 1 from the nth predicted index 
					self.alpha[activation][i] -= 1 

			print ("Training mistakes")
			print mistake
			self.mistakes_train.append(mistake)
			self.classifier_test()

		
		return self.alpha


	def classifier_test(self):
		# for testing set
		mistake = 0

		for i in range (len(self.test_list_word)):
			A = []
			a = 0
			for j in range (len (self.alpha)):

				for k in range (len (self.alpha[j])):

					if self.alpha[j][k] != 0:

						a_ = np.dot(self.train_list_word[k], self.test_list_word[i]) + 1
						a__ = math.pow(a_, 2)
						# print self.alpha[j][k]
						a += self.alpha[j][k] * a__ 

				A.append(a)
			
			activation = np.argmax(A)
			
			if  activation != self.test_list_letter[i]:
				mistake += 1

		print ("Test Mistake")
		print mistake
		self.mistakes_test.append(mistake)

			


def main():
	path1 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_train.txt"
	path2 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_test.txt"

	classifier = KernelizedPerceptron_multi (path1, path2)
	a = classifier.classifier_train()
	# print a
	print classifier.mistakes_train
	print classifier.mistakes_test
	print len(classifier.train_list_word)
	print len(classifier.test_list_word)
	plt.plot(classifier.mistakes_train)
	plt.plot(classifier.mistakes_test)
	plt.show()




	# plots the testing accuracy of the dataset
	# plt.plot(accuracy_validation)

	plt.xlabel("Number of Iterations", fontsize = 15)
	plt.ylabel(" Number of Mistakes", fontsize = 15)
	plt.title("Mistakes (Kernelized Perceptron(Multi))", fontsize = 25)
	# plt.ylim([0.1, 0.8])
	plt.grid(True)
	# plt.legend(['Training', ' Testing', 'Validation'])
	plt.show()  



if __name__ == '__main__':

	main()

