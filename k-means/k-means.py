import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

N_CLUSTERS = 20

myfile = open("lineNtest.csv", "r")
d = []
for l in myfile:
    l.rstrip('\n')
    x = l.split(',')
    d.append([float(x[0]), float(x[1])])
npd = np.array(d)

plt.scatter(npd[:,0],npd[:,1],s=18)
plt.grid()
plt.show()

cls = KMeans(n_clusters=N_CLUSTERS)
result = cls.fit(npd)

plt.scatter(npd[:,0],npd[:,1],s=18, c=result.labels_)
plt.scatter(result.cluster_centers_[:,0],result.cluster_centers_[:,1],s=70, marker='*',c='red')
plt.show()