import numpy as np
import math

# Calculate no. of cycles for 3 dataflows
# input stationary, weight stationary, row stationary
# Account for tiling
# Given, convolution operation for input size (X*Y), filter size (R*S), no. of. channels (C), no. of filters (K)
# Systolic array size (array_height * array_width)
# Complete the functions as per the instructions
# Assume no padding
# Assume horizontal strides = vertical strides


class dataflow:
    def __init__(self, X, Y, R, S, C, K, strides, array_height, array_width):
        self.X = X
        self.Y = Y

        self.R = R
        self.S = S

        self.C = C
        self.K = K

        self.array_height = array_height
        self.array_width = array_width
        self.strides = strides

        self.rs_compute_cycles = 0
        self.ws_compute_cycles = 0
        self.is_compute_cycles = 0
        self.os_compute_cycles = 0

        output_h = math.floor((self.Y - self.S + self.strides) / self.strides)
        output_w = math.floor((self.X - self.R + self.strides) / self.strides)
        self.w_conv = self.C * self.R * self.S
        self.n_ofmap = output_h * output_w

    # For row stationary, following parameters are constant
    # R = S = array_height = array_width = 3
    def row_stationary(self):
        # Write your code here
        self.rs_compute_cycles = math.ceil(self.Y / 5) * (self.S + self.R + 3 + self.X - 2) * self.C * self.K
        return self.rs_compute_cycles

    def input_stationary(self):
        # Write your code here
        S_R = self.w_conv
        S_C = self.n_ofmap
        T = self.K
        self.is_compute_cycles = (
            (2 * self.array_height + self.array_width + T - 2)
            * math.ceil(S_R / self.array_height)
            * math.ceil(S_C / self.array_width)
        ) - 1
        return self.is_compute_cycles

    def weight_stationary(self):
        # Write your code here
        S_R = self.w_conv
        S_C = self.K
        T = self.n_ofmap
        self.ws_compute_cycles = (
            (2 * self.array_height + self.array_width + T - 2)
            * math.ceil(S_R / self.array_height)
            * math.ceil(S_C / self.array_width)
        ) - 1
        return self.ws_compute_cycles
