"""
MIT License:
Copyright (c) 2023 Muhammad Umer

BPST Transmitter
"""

import numpy as np

from .helper import *


class BPSKTransmitter:
    """A class used to represent a BPSK Transmitter"""

    def __init__(self, gain=10):
        """
        Initialize the BPSK Transmitter.

        Args:
            channel_gain (float, optional): Gain of the channel in dB.
        """
        self.channel_gain = 10 ** (gain / 10)

    def to_arr(self, binary_representation):
        """
        Convert binary representation to numpy array.

        Args:
            binary_representation (str): Binary string to be converted.

        Returns:
            binary_array (np.array): Numpy array of binary digits.
        """

        return to_arrayC(binary_representation)  # type: ignore

    def modulate(self, binary_array):
        """
        Modulate the binary array using BPSK modulation.

        Args:
            binary_array (np.array): Numpy array of binary digits.

        Returns:
            bpsk_modulated (np.array): BPSK modulated signal.
        """
        bpsk_modulated = np.where(binary_array == 0, -1 + 0j, 1 + 0j)
        return bpsk_modulated

    def apply_channel(self, bpsk_modulated):
        """
        Apply channel effects to the BPSK modulated signal.

        Args:
            bpsk_modulated (np.array): BPSK modulated signal.
            channel_gain (float, optional): Gain of the channel. Defaults to 10 ** (10 / 10).

        Returns:
            received_signal (np.array): Signal received after channel effects.
        """

        raise NotImplementedError

    def demodulate(self, received_signal):
        """
        Demodulate the received signal.

        Args:
            received_signal (np.array): Signal received after channel effects.

        Returns:
            bpsk_demodulated (np.array): Demodulated BPSK signal.
        """
        bpsk_demodulated = np.where(received_signal.real < 0, 0, 1)
        return bpsk_demodulated

    def concatenate_bits(self, bpsk_demodulated):
        """
        Concatenate the demodulated bits.

        Args:
            bpsk_demodulated (np.array): Demodulated BPSK signal.

        Returns:
            received_bits (np.array): Concatenated bits from the demodulated signal.
        """

        return concatenate_bitsC(bpsk_demodulated)  # type: ignore

    def forward(self, binary_representation):
        """
        Forward method to process the binary representation through the transmitter.

        Args:
            binary_representation (str): Binary string to be processed.

        Returns:
            received_bits (np.array): Concatenated bits from the processed signal.
        """
        binary_array = self.to_arr(binary_representation)
        bpsk_modulated = self.modulate(binary_array)
        received_signal = self.apply_channel(bpsk_modulated)
        bpsk_demodulated = self.demodulate(received_signal)
        received_bits = self.concatenate_bits(bpsk_demodulated)
        return received_bits
