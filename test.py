import numpy as np
from clustering.equal_groups import EqualGroupsKMeans

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

clf = EqualGroupsKMeans(n_clusters=2)

clf.fit(X)

clf.labels_