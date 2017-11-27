import matplotlib.pyplot as plt


# x = [0.001, 0.01, 0.1, 1, 10]

x1 = [1056./2000., 1126./2000., 1135./2000., 1147./2000., 1137./2000., 1135./2000.]
x2 = [1198./2000., 1295./2000., 1522./2000, 1547./2000, 1569./2000, 1619./2000]
x3 = [1181./2000., 1243./2000., 1402/2000., 1430/2000., 1394/2000., 1428/2000.]
x4 = [1295/2000., 1295/2000., 1523/2000., 1548/2000., 1567/2000., 1579/2000.]
x5 = [1198/2000., 1295/2000., 1522/2000., 1547/2000., 1569/2000., 1619/2000.]
plt.plot(x1, marker = 'o')
plt.plot(x2, marker = '<')
plt.plot(x3, marker = '>')
plt.plot(x4, marker = 'v')
plt.plot(x5, marker = '^')

plt.xlabel("Number of trees", fontsize = 15)
plt.ylabel(" Accuracy", fontsize = 15)
plt.title("DT (Bagging) Testing", fontsize = 25)
plt.grid(True)
plt.legend(['Depth = 1', 'Depth = 2', 'Depth = 3', 'Depth = 4', 'Depth = 5'])


plt.show()


