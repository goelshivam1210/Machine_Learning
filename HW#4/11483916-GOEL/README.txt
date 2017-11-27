README

General information

The code implements the GMM for a one dimensional data and the code is written in python 2.7.12
The code is well commented for use.
The main function runs the algorithm for different values of k = (3, 4, 5)
the variable path takes in the name of the data file.
It spits out the value of the likelihood upon convergence.
It also spits out the parameters for each k 

Libraries
The code uses the library numpy





parameters

k = 3

loglikelihood   -10443.0268216
variance            [ 0.82047858  5.01874637  0.45023265]
mean               [ 5.50927837  20.46790871   5.5977876 ]
prior            [3.33333209e-01   6.66666581e-01   1.07300449e-09]

k = 4

loglikelihood -10385.0294192
variance            [ 4.96994457  0.43223696  0.43140235  0.7876642 ]
mean               [ 10.47922651  15.47539173  15.44073061  25.49079987]
prior            [ 6.66667009e-01   2.24843089e-10   5.62909223e-11   3.32896871e-01]

loglikelihood  -10385.0288678
variance          [ 4.96994458  0.43154755  0.78766419  0.43173291  0.43181301]
mean 			  [ 10.47922671  15.45061622  25.49079988  15.45897877  15.46285607]
prior			  [  6.66667034e-01   5.10593426e-11   3.32896869e-01   1.90223675e-11    1.22290885e-11]


