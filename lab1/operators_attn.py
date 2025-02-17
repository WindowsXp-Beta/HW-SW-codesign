from operator_base import Operator
import numpy as np

class Logit(Operator):
    def __init__(self, dim):
        super().__init__(dim=dim)
        
        self.op_type = 'Logit'

    def get_effective_dim_len(self):
        return 5

    def get_tensors(self):
        B, H, M, N, D = self.dim[:self.get_effective_dim_len()]
        input_a = B * H * M * D
        input_w = B * H * N * D
        output = B * H *  M * N
        return input_a, input_w, output

    def get_num_ops(self):
        B, H, M, N, D = self.dim[:self.get_effective_dim_len()]
        return np.prod([B, H, M, N, D])

class Attend(Operator):
    def __init__(self, dim):
        super().__init__(dim=dim)
        self.op_type = 'Attend'
        
    def get_effective_dim_len(self):
        return 5

    def get_tensors(self):
        B, H, M, N, D = self.dim[:self.get_effective_dim_len()]
        input_a = B * H *  M * N
        input_w = B * H * N * D
        output = B * H * M * D
        return input_a, input_w, output

    def get_num_ops(self):
        B, H, M, N, D = self.dim[:self.get_effective_dim_len()]
        return np.prod([B, H, M, N, D])
