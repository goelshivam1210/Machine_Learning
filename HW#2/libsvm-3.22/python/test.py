import matplotlib.pyplot as plt


# x = [0.001, 0.01, 0.1, 1, 10]

x1 = [338./600., 314./600., 351./600., 360./600., 359./600., 351./600.]
x2 = [384./600., 405./600., 487./600, 532./600, 542./600, 557./600]
x3 = [380./600., 373/600., 435/600., 438/600., 433/600., 441/600.]
x4 = [384/600., 405/600., 483/600., 510/600., 512/600., 516/600.]
x5 = [384/600., 405/600., 487/600., 532/600., 542/600., 557/600.]
plt.plot(x1)
plt.plot(x2)
plt.plot(x3)
plt.plot(x4)
plt.plot(x5)

plt.xlabel("Number of trees", fontsize = 15)
plt.ylabel(" Accuracy", fontsize = 15)
plt.title("DT(Bagging) Training", fontsize = 25)
plt.grid(True)
plt.legend(['Depth = 1', 'Depth = 2', 'Depth = 3', 'Depth = 4', 'Depth = 5'])


plt.show()


