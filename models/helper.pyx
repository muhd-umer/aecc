# distutils: language = c++
# cython: language_level=3
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np

def to_arrayC(np.ndarray binary_data):
    # Define a 4D numpy array
    cdef np.ndarray[np.int_t, ndim=4] binary_array

    # Convert binary data to a 4D numpy array
    binary_array = np.array(
        [
            [
                [np.array(list(map(int, binary_str))) for binary_str in binary_row]
                for binary_row in binary_matrix
            ]
            for binary_matrix in binary_data
        ],
        dtype=int
    )

    # Return the 4D numpy array
    return binary_array

def demodulateC(np.ndarray[np.complex64_t, ndim=4] received_signal):
    # Perform BPSK demodulation in-place
    np.where(np.real(received_signal) < 0, 0, 1, out=received_signal)
    return received_signal.astype(int)

def to_binaryC(np.ndarray[np.int_t, ndim=4] binary_array):
    # Convert 4D numpy array to binary data in-place
    for i in range(binary_array.shape[0]):
        for j in range(binary_array.shape[1]):
            for k in range(binary_array.shape[2]):
                for l in range(binary_array.shape[3]):
                    binary_array[i][j][k][l] = str(binary_array[i][j][k][l])

    # Return the binary data
    return binary_array