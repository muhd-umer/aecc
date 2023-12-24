"""
MIT License:
Copyright (c) 2023 Muhammad Umer

IEEE 754 Floating Point Converter
"""

import numpy as np
from numba import vectorize


class IEEE754:
    """A class to represent the IEEE-754 floating point standard."""

    def __init__(self, precision="single"):
        """Initializes the IEEE 754 object with the specified precision.

        Args:
            precision (str): The desired precision, either "single" or "double".
        """
        if precision != "single":
            raise NotImplementedError("Only single precision is currently supported.")
        self.precision = precision

    @staticmethod
    @vectorize
    def float_to_int(float_value):
        return np.float32(float_value).view(np.uint32)

    @staticmethod
    @vectorize
    def int_to_float(integer_value):
        return np.int32(integer_value).view(np.float32)

    def float_to_binary(self, float_value):
        """Converts a float to its unsigned binary representation using IEEE 754.

        Args:
            float_value (float): The float value to convert.

        Returns:
            str: The unsigned binary representation of the float value.
        """
        integer_part = self.float_to_int(float_value)
        binary_string = np.vectorize(lambda x: f"{x:032b}")(integer_part)
        return binary_string

    def binary_to_float(self, binary_string):
        """Converts an unsigned binary string to its float representation using IEEE 754.

        Args:
            binary_string (str): The unsigned binary string to convert.

        Returns:
            float: The float value represented by the binary string.
        """
        integer_part = np.vectorize(lambda x: int(x, 2))(binary_string)
        float_value = self.int_to_float(integer_part)
        return float_value
