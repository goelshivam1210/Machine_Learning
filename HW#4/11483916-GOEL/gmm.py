#gmm.py
'''
author: Shivam Goel
WSUID# 11483916
Reference 
lecture notes
ciml.info
https://www.youtube.com/watch?v=iQoXFmbXRJA


'''
import numpy as np
import math
import random


class GMM():

	def __init__(self, path, k):
		#k is the number of clusters
		self.k = k
		self.x = self.parser(path)
		# means of the cluster
		self.mu = np.empty(shape = int(self.k), dtype = float)
		# variances of the clusters
		self.sigma = np.empty(shape = int(self.k), dtype = float)
		# prior probability of each cluster
		self.theta = np.empty(shape = int(self.k), dtype = float)
		# assignments
		# self.zed = for i in range(0, int(len(self.k)))
		self.zed = np.empty(shape = (len(self.x), int(self.k)), dtype = float)
		# for saving unnormalized z to calculate the log likelihood
		self.zed_unormal = np.empty(shape = (len(self.x), int(self.k)), dtype = float)
		# list of the log likelihood at each iteration
		self.likelihood  = [0]


	def parser(self, path1):
		x = []
		f = open(path1)
		for line in f :
			line = line.strip()
			x.append(line)
		return np.array(x, dtype = float)

	def performclustering(self):

		for i in range(int(self.k)):
			# random initialize the mean for the clusters
			self.mu[i] = random.choice(self.x)
			# initialze the variances to 1
			self.sigma[i] = 1
			# initialize the prior probability of each cluster to be equally likely
			self.theta[i] = float(1/self.k)
		# loop for iterations
		while True:
			# for the fractional assignments
			for i in range(len(self.x)):

				for j in range(int(self.k)):
			
					# calculating the fractional assignemnts for each data point in each cluster
					# print self.sigma[j]
					zed3 = 2* math.pi * self.sigma[j]
					zed1 = np.power(zed3, -0.5)
					zed2 = np.exp((-1/(2*(self.sigma[j]))) * ((self.x[i] - self.mu[j])**2))
					self.zed[i][j] = self.theta[j]  * zed1 * zed2
					# print self.zed[i][j]
					# time.sleep(0.001)
					# save the unormalized zed for calculation of log likelihood
					self.zed_unormal[i][j] = self.theta[j]  * zed1 * zed2
					# print self.zed_unormal[0]
			
				# normalizing the vector
				for j in range(int(self.k)):
					self.zed[i][j] *=  1/np.sum(self.zed[i])
			# calculate the log likeihood of z
			loglikelihood = 0
			for i in range(len(self.x)):
					loglikelihood += np.log10(np.sum(self.zed_unormal[i]))
			self.likelihood.append(loglikelihood)

			# compute the sumof array using axis and use that sum later for summations
			sumofarray = np.sum(self.zed, axis = 0)

			for i in range(int(self.k)):

				# recompute the prior probability of the cluster

				self.theta[i] = sumofarray[i]/len(self.x)
				
				# recompute the mean of each cluster
				sum2 = 0
				for j in range(len(self.x)):
					sum2 += self.zed[j][i] * self.x[j]
				
				self.mu[i] =  sum2/sumofarray[i]

				# recompute the variance of each cluster
				sum2 = 0
				for j in range(len(self.x)):

					sum2 += self.zed[j][i] * (abs(self.x[j] - self.mu[i]))
					# print self.zed[][]
					# print sum2

				self.sigma[i] = sum2/sumofarray[i]

			
			length = len(self.likelihood)
			if abs(self.likelihood[length-2] - self.likelihood[length-1]) < 0.01:
				print self.likelihood[length -1]
				break
		return self.sigma, self.mu, self.theta


def main():
	path = "em_data.txt"
	K = [3., 4., 5.]
	for i in range(len(K)):
		cluster = GMM(path, K[i])
		a, b, c = cluster.performclustering()
		print ("Parameters for K = {}".format(K[i]))
		print ("Variance   {}".format(a))
		print ("Mean   {}".format(b))
		print ("Prior Probability   {}".format(c))


if __name__ == '__main__':

	main()

	

			# calculate the probability that a  
		# lets assume there are k gaussians as the data has k clusters..
		