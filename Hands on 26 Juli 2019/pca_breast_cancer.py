# PCA

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)
centers = [[1, 1], [-1, -1], [1, -1]]
breastcancer = datasets.load_breast_cancer()
x = breastcancer.data
y = breastcancer.target

fig = plt.figure(1, figsize = (4, 3))
plt.clf()
ax = Axes3D(fig, rect = [0, 0, 0.95, 1], elev = 48, azim = 134)

plt.cla()
pca = decomposition.PCA(n_components = 2) # Mengambil tiga fitur dengan mengambil transformer PCA
pca.fit(x) # mencari matriks w transpose
x = pca.transform(x) # mengubah x menjadi z. ingat rumus z = wtranspose * x

for name, label in [('Malignant', 0), ('Benign', 1)]:
    ax.text3D(x[y == label, 0].mean(),
              x[y == label, 1].mean() + 1.5, name,
              horizontalalignment = 'center',
              bbox = dict(alpha = 0.5, edgecolor = 'w', facecolor = 'w'))

y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(x[:, 0], x[:, 1], c = y, cmap = plt.cm.nipy_spectral, edgecolor = 'k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()

# Mengamati seberapa besar PoV-nya.
# Fitur pertama 92,5%, fitur kedua 5,3%, fitur ketiga 1,7%
print("Explained variance ratio (first 2 components): %s" %str(pca.explained_variance_ratio_))