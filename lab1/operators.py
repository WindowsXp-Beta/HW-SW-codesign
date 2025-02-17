from operator_base import Operator

# The last digit of my GT ID is 9
ADV_OP_CNT = 9 + 3

class SoftMax(Operator):
    # [B, L] -> SoftMax -> [B, L]
    def __init__(self, dim):
        super().__init__(dim=dim)
        self.op_type = 'SoftMax'

    def get_effective_dim_len(self):
        return 2

    ################################################################
    ## TODO 1.A
    ################################################################
    def get_tensors(self):
        # Function Objective = Derive the number of input & output elements in the SoftMax operation.
        # Use 2 dimensions of SoftMax inputs to derive the values.
        # Function  Output = Number of elements of input_a  and output
        B, L = self.dim[:self.get_effective_dim_len()]
        input_a = B * L
        input_b = 0 # Since single input operation, we will keep this 0.
        output = B * L
        return input_a, input_b, output

    ################################################################
    ## TODO 1.A
    ################################################################
    def get_num_ops(self):
        # Function Objective = Derive the number of individual operations in SoftMax.
        # Consider exponential function (e^x) as (GTID+3) operations and division as (GTID+3) operations too
        # Use 2 dimensions of SoftMax inputs to derive the values.
        # Function Output = Number of operations in SoftMax
        B, L = self.dim[:self.get_effective_dim_len()]
        sum_ops = L * ADV_OP_CNT + (L - 1) * 1
        num_ops = B * (sum_ops + L * ADV_OP_CNT)
        return num_ops

class layer_norm(Operator):
    # [B, X, Y] -> [B, X, Y]

    def __init__(self, dim):
        super().__init__(dim=dim)
        self.op_type = 'layer_norm'

    def get_effective_dim_len(self):
        return 3

    ################################################################
    ## TODO 1.B
    ################################################################
    def get_tensors(self):
        # Function Objective = Derive the number of input & output elements in the layer normalization operation.
        # Use dimensions of the tensor:  batch, tensor height and tensor width to derive the values.
        # Function Output = Number of elements of input_a and output
        B, X, Y = self.dim[:self.get_effective_dim_len()]
        input_a = B * X * Y
        input_b = 0 # Since single input operation, we will keep this 0.
        output = B * X * Y
        return input_a, input_b, output

    ################################################################
    ## TODO 1.B
    ################################################################
    def get_num_ops(self):
        ## Function get_num_ops for layer normalization.
        # Substraction is considered one operation while division is considered (GT_ID+3) operations and square root is considered (GT_ID+3) operations as well
        # Also, count the number of operations to calculation mean and standard deviation
        # Use dimensions of the tensor:  batch , tensor height and tensor width to derive the values.
        # Function  Output = Total number of individual operation in the given layer normalization
        B, X, Y = self.dim[:self.get_effective_dim_len()]
        mean_ops = (Y - 1) * 1 + ADV_OP_CNT
        dev_ops = Y * (1 + 1) + (Y - 1) * 1 + ADV_OP_CNT * 2
        num_ops = B * X * (mean_ops + dev_ops + Y * (1 + ADV_OP_CNT))
        return num_ops

class GEMM(Operator):
    # [B, M, K] * [K, N] = [B, M, N]
    def __init__(self, dim):
        super().__init__(dim=dim)
        self.op_type = 'GEMM'

    def get_effective_dim_len(self):
        return 4


    ################################################################
    ## TODO 1.C
    ################################################################
    def get_tensors(self):
        # Function Objective = Derive the number of input & output elements in the multiplication of 2 matrix.
        # Use dimensions of the matmul to derive the values.
        # A x B = C; Dim (A) = [B , M, K], Dim (B) = [K, N], Dim (C) = [B, M, N], Reduction dimension = K
        # Function Output = Number of elements of input_a , input_b and output
        B, M, N, K = self.dim[:self.get_effective_dim_len()]
        input_a = B * M * K
        input_b = K * N
        output = B * M * N
        return input_a, input_b, output


    ################################################################
    ## TODO 1.C
    ################################################################
    def get_num_ops(self):
        # Function Objective = Derive the number of operations (multiplication + additions)
        # Use dimensions of the matmul to derive the values.
        # A x B = C; Dim (A) = [B , M, K], Dim (B) = [K, N], Dim (C) = [B, M, N], Reduction dimension = K
        # Function Output = Total number of individual operation in the given Matmul.
        B, M, N, K = self.dim[:self.get_effective_dim_len()]
        # Addition of each element is consider a single operation, i.e. 8+8+0+3 = 3 operations.
        # Multiplication of each element is consider a single operation, i.e. 2x4 = 1 operation.
        # Ex: Matrix of 1x100 and 100x1 will require 100 multiplications and 99 additions, so total no. of operations is 100 + 99 = 199
        num_ops = B * (2 * K - 1) * M * N
        return num_ops


class attention(Operator):
    # Input:[B, M, K], Wq/Wk/Wv: [B, K, N], Output [B, M, N]
    def __init__(self, dim):
        super().__init__(dim=dim)
        self.op_type = 'attention'

    def get_effective_dim_len(self):
        return 4

    ################################################################
    ## TODO 1.D
    ################################################################
    def get_tensors(self):
        ## Function get_tensors for attention.
        # Function Objective = Derive the number of elements of inputs and outputs in the scaled dot product attention operator
        # Use dimensions of the input and weights to derive the values.
        # Q = X*Wq; K = X*Wk, V = X*Wv, Output = SoftMax(Q*K^T/(sqrt(c)))*V
        # Function  Output = Number of elements of input_a , input_b (sum of of Wq, Wk, Wv elements) and output
        B, M, N, K = self.dim[:self.get_effective_dim_len()]
        input_a = B * M * K
        input_b = 3 * K * N
        output = B * M * N
        return input_a, input_b, output

    ################################################################
    ## TODO 1.D
    ################################################################
    def get_num_ops(self):
        ## Function get_num_ops for attention
        # Function Objective = Derive the number of operations (you may reuse no. of operations in softmax and gemm)
        # Use dimensions of the input and weights to derive the values.
        # Q = X*Wq; K = X*Wk, V = X*Wv, Output = SoftMax(Q*K^T/(sqrt(c)))*V
        # Function  Output = Total number of operations in the attention operator
        B, M, N, K = self.dim[:self.get_effective_dim_len()]
        qkv_construct_ops = 3 * (2 * K - 1) * M * N
        qkt_ops = (2 * N - 1) * M * M
        inter_ops = M * M * (ADV_OP_CNT) + ADV_OP_CNT
        softmax_ops = M * ((M * ADV_OP_CNT + (M - 1)) + M * ADV_OP_CNT)
        inter_v_ops = (2 * M - 1) * M * N
        num_ops = B * (qkv_construct_ops + qkt_ops + inter_ops + softmax_ops + inter_v_ops)
        return num_ops