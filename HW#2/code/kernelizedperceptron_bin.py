from svmutil import *
import numpy as np
import math
import random
import matplotlib.pyplot as plt


class KernelizedPerceptron_binary():

	def __init__(self, path1, path2):

		self.train_list_letter, self.train_list_word = self.parse(path1)
		self.test_list_letter, self.test_list_word = self.parse(path2)


		self.alpha = np.zeros(len(self.train_list_word), dtype = np.float32)
		self.bias = 0
		self.maxiter = 20

		self.mistakes_train = np.zeros(self.maxiter, dtype = np.int32)
		self.mistakes_test = np.zeros(self.maxiter, dtype = np.int32)


	def parse(self, file):

		# list of the x's
		list_word = []
		# list of the y's
		list_letter = []
		#holds the whole list
		#list_main = []

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
				#word = (map(float, word))
				word = [float(w) for w in word]
				if letter == 0 or letter == 4 or letter == 8 or letter == 14 or letter == 20:
					letter = 1
				else:
					letter = -1
				# print word
				# if validation set is being computed
				list_word.append(word)
				list_letter.append(letter)
				#list_main.append(g)
		
		return np.array(list_letter, dtype=int), np.array(list_word, dtype=int) #, list_main


	def classifier_train (self):

		for iter in range (self.maxiter):
			# print (iter)
			mistake = 0

			for i in range (len(self.train_list_word)):
				# now we have to compute the activation function
				# computation of activation function
				a = 0
				for j in range (len (self.alpha)):
					a_ = np.dot(self.train_list_word[j], self.train_list_word[i]) + 1

					a__ = math.pow(a_, 2)

					a += self.alpha[j] * a__ + self.bias


				# condition for the mistake check
				if self.train_list_letter[i] * a <= 0:
					
					mistake += 1
					# update the alphas
					self.alpha[i] += self.train_list_letter[i]
					# update the bias
					self.bias += self.train_list_letter[i]

			self.mistakes_train[iter] = mistake
			# perform the testing after one iteration of training
			self.mistakes_test[iter] = self.classifier_test()

		
		return self.alpha, self.bias


	def classifier_test(self):
		# for testing set
		mistake = 0
		# print ("testing")
		for i in range (len(self.test_list_word)):

			a =0
			for j in range (len(self.alpha)):
			
				a_ = np.dot(self.train_list_word[j], self.test_list_word[i]) + 1

				a__ = math.pow(a_, 2)

				a += self.alpha[j] * a__ + self.bias

			if self.test_list_letter[i] * a <= 0:

				mistake += 1
		return mistake

			


def main():
	path1 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_train.txt"
	path2 = "/home/goelshivam12/Desktop/ML_Homework/HW#1/OCRdata/ocr_fold0_sm_test.txt"

	classifier = KernelizedPerceptron_binary(path1, path2)
	a, b = classifier.classifier_train()
	# print a
	print (classifier.mistakes_train)
	print (classifier.mistakes_test)
	print (len(classifier.train_list_word))
	print (len(classifier.test_list_word))

	plt.plot(classifier.mistakes_train)
	plt.plot(classifier.mistakes_test)
	plt.show()

	# plots the testing accuracy of the dataset
	# plt.plot(accuracy_validation)

	# plt.xlabel("C's", fontsize = 15)
	# plt.ylabel(" Accuracy", fontsize = 15)
	# plt.title("Accuracy Curve (SVM)", fontsize = 25)
	# # plt.ylim([0.1, 0.8])
	# plt.grid(True)
	# plt.legend(['Training', ' Testing', 'Validation'])
	# plt.show()  



if __name__ == '__main__':

	main()


		# compute the training classifier and each time please test its accuracy on the validation data as well as the test data 




