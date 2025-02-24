from dataflow import dataflow

X = 40
Y = 40

R = 3
S = 3

C = 3
K = 16

strides = 1

array_height = 4
array_width = 3




dataflow = dataflow(X, Y, R, S, C, K, strides, array_height, array_width)


print(dataflow.row_stationary())
print(dataflow.input_stationary())
print(dataflow.weight_stationary())
