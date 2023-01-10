import numpy as np

x = np.random.randint(10,50,110).reshape(11,10) # 11 samples, 10 features each

# mean centering the data
x_meaned = x - np.mean(x, axis=0) # axis=0 because for each sample (from the 11) we calculate the mean of
# each feature (feature = row)

print(x_meaned.shape)
cov_mat = np.cov(x_meaned, rowvar=False)

# calculating eigenvalues and eigenvectors of the covariance matrix
eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

# the eigen vectors are the principal components. a higher eigenvalue corresponds to a higher variability.
# the principal axis with the higher variability will be an axis capturing higher variability in data.
# we sort them in descending order -> first eigen vectors will be the principal components
# that capture higher variability.

# sort eigen values in descending order
sorted_index = np.argsort(eigen_values)[::-1]

sorted_eigenvalues = eigen_values[sorted_index]
# sort eigen vectors
sorted_eigenvectors = eigen_vectors[:,sorted_index]

# select the first n eigenvectors, n is desired dimension
# of our final reduced data.
n_components = 2
eigenvector_subset = sorted_eigenvectors[:, 0:n_components]

# we transform the data by having the dot product between the transpose of the eigenvector
# subset and the transpose of the mean-centered data. by transposing the outcome of the dot
# product, the result we get is the data reduced to lower dimensions from higher dimensions
x_reduced = np.dot(eigenvector_subset.transpose(), x_meaned.transpose()).transpose()

print(x_reduced.shape)
