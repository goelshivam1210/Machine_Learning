import matplotlib.pyplot as plt


# x = [0.001, 0.01, 0.1, 1, 10]

x1 = [376./600., 376./600., 376./600., 376./600., 376./600., 376./600.]
x2 = [426./600., 449./600., 449./600, 449./600, 449./600, 449./600]
x3 = [498./600., 518./600., 532/600., 532/600., 532/600., 532/600.]

plt.plot(x1, marker = 'o')
plt.plot(x2, marker = '<')
plt.plot(x3, marker = '>')

plt.xlabel("Number of Boosting iterations", fontsize = 15)
plt.ylabel(" Accuracy", fontsize = 15)
plt.title("DT (Boosting) Training", fontsize = 25)
plt.grid(True)
plt.legend(['Depth = 1', 'Depth = 2', 'Depth = 3'])


plt.show()


