# distutils: language = c++
# cython: language_level=3
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np

# Function to convert binary data to a 4D numpy array
def to_arrayC(list binary_data):
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

# Function to concatenate bits from a demodulated BPSK signal
def concatenate_bitsC(list bpsk_demodulated):
    # Define a 3D numpy array
    cdef np.ndarray[object, ndim=3] received_bits

    # Concatenate bits from a demodulated BPSK signal
    received_bits = np.array(
        [
            [
                [''.join(str(bit) for bit in binary_array) for binary_array in binary_row]
                for binary_row in binary_matrix
            ]
            for binary_matrix in bpsk_demodulated
        ],
        dtype=object
    )

    # Return the 3D numpy array
    return received_bits