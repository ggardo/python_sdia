import numpy as np
cimport numpy as np
np.import_array()

import bottleneck as bn

def KNN_cy(np.ndarray[np.float64_t, ndim=2] x_train, np.ndarray[np.float64_t, ndim=1] class_train,
            np.ndarray[np.float64_t, ndim=2] x_test, int n_neighbours):

    cdef Py_ssize_t N_train = x_train.shape[0]
    cdef Py_ssize_t N_test = x_test.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] distances = np.zeros((N_test, N_train), dtype=np.float64)  # Specify dtype
    cdef int i
    for i in range(N_test):
        # Compute the distance between each row of x_test and x_train
        distances[i, :] = np.linalg.norm(x_train - x_test[i, :], axis=1)
    # Sort the distances and keep the indices of the nearest neighbors
    cdef np.ndarray[long, ndim=2] id = bn.argpartition(distances, n_neighbours - 1, axis=1)[:, :n_neighbours]
    # Retrieve the labels of the nearest neighbors
    cdef np.ndarray[np.float64_t, ndim=2] labels = class_train[id]
    cdef np.ndarray[np.float64_t, ndim=1] class_pred = np.zeros(N_test)
    for i in range(N_test):
        # Give to class_pred the most frequent label among the nearest neighbors
        class_pred[i] = np.argmax(np.bincount(labels[i, :].astype(int)))
    return class_pred