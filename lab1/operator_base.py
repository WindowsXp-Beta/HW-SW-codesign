import numpy as np
from unit import Unit


class Operator(object):
    def __init__(self, dim):
        self.dim = dim
        self.input_a, self.input_w, self.output = self.get_tensors()
        self.num_ops = self.get_num_ops()

    def set_tensor(self, input_a=None, input_w=None, output=None):
        if input_a is not None:
            self.input_a = input_a
        if input_w is not None:
            self.input_w = input_w
        if output is not None:
            self.output = output


    def get_op_type(self):
        return self.op_type

    def get_tensors(self):
        pass

    def get_num_ops(self):
        pass

    def get_effective_dim_len(self):
        pass


    ################################################################
    ## TODO A.2.i
    ################################################################
    # Use number of operations for given operator and system parameters to determine the compute time.
    def get_ideal_compute_time(self, system, data_format):
        number_of_ops = self.get_num_ops()
        flop_fac = {
            "bf16": 2,
            "int8": 4,
            "fp32": 1,
            "fp64": 0.5
        }[data_format]
        compute_time = number_of_ops / (system.op_per_sec * flop_fac) / system.compute_efficiency
        return compute_time

    ################################################################
    ## TODO A.2.ii
    ################################################################
    # Use number of elements for given operator and system parameters to determine the memory time.

    def get_ideal_memory_time(self, system, data_format):
        ## Number of elements
        input_a, input_b, output = self.get_tensors()
        ## Assume data format of FP32 for all both inputs and outputs.
        bytes_per_element = {
            "fp32": 4,
            "bf16": 2,
            "int8": 1,
            "fp64": 8
        }[data_format]
        mem_txn_times = []
        mem_txn_times.append(input_a * bytes_per_element / system.offchip_mem_bw / system.memory_efficiency)
        mem_txn_times.append(input_b * bytes_per_element / system.offchip_mem_bw / system.memory_efficiency)
        mem_txn_times.append(output * bytes_per_element / system.offchip_mem_bw / system.memory_efficiency)
        memory_total_time = sum(mem_txn_times)
        return memory_total_time

    # def get_ideal_memory_time(self, system, tiling, Tm, Tk, Tn):
    #     ## Number of elements
    #     input_a, input_b, output = self.get_tensors()
    #     ## Assume data format of FP32 for all both inputs and outputs.
    #     bytes_per_element = 4
    #     mem_txn_times = []
    #     if tiling == "A":
    #         f_a, f_b, f_c = 1, Tn, 2 * Tn
    #     elif tiling == "B":
    #         f_a, f_b, f_c = Tm, 1, 2 * Tm
    #     else:
    #         f_a, f_b, f_c = Tk, Tk, 2
    #     mem_txn_times.append(f_a * input_a * bytes_per_element / system.offchip_mem_bw / system.memory_efficiency)
    #     mem_txn_times.append(f_b * input_b * bytes_per_element / system.offchip_mem_bw / system.memory_efficiency)
    #     mem_txn_times.append(f_c * output * bytes_per_element / system.offchip_mem_bw / system.memory_efficiency)
    #     memory_total_time = sum(mem_txn_times)
    #     return memory_total_time

    # def get_ideal_memory_time(self, system, fusion_pos):
    #     ## Number of elements
    #     input_a, input_b, output = self.get_tensors()
    #     ## Assume data format of FP32 for all both inputs and outputs.
    #     bytes_per_element = 4
    #     mem_txn_times = []
    #     if (fusion_pos == 0 or fusion_pos == -2):
    #         mem_txn_times.append(input_a * bytes_per_element / system.offchip_mem_bw / system.memory_efficiency)
    #     mem_txn_times.append(input_b * bytes_per_element / system.offchip_mem_bw / system.memory_efficiency)
    #     if (fusion_pos == -1 or fusion_pos == -2):
    #         mem_txn_times.append(output * bytes_per_element / system.offchip_mem_bw / system.memory_efficiency)
    #     memory_total_time = sum(mem_txn_times)
    #     return  memory_total_time


    def get_roofline(self, system, data_format):
    # def get_roofline(self, system, tiling, Tm, Tk, Tn):
    # def get_roofline(self, system, fusion_pos):
        unit = Unit()
        # ideal_compute_time = self.get_ideal_compute_time(system=system)
        ideal_compute_time = self.get_ideal_compute_time(system=system, data_format=data_format)
        # ideal_memory_time = self.get_ideal_memory_time(system=system, fusion_pos=fusion_pos)
        # ideal_memory_time = self.get_ideal_memory_time(system=system, tiling=tiling, Tm=Tm, Tk=Tk, Tn=Tn)
        ideal_memory_time = self.get_ideal_memory_time(system=system, data_format=data_format)
        num_ops = self.get_num_ops()
        input_a_size, input_w_size, output_size = self.get_tensors()

        num_data = (input_a_size + input_w_size + output_size)
        op_intensity = num_ops/num_data

    ################################################################
    ## TODO A.2.iii
    ################################################################
    # Assume the computation and memory operation is perfectly synchronized  so they can be executed in parallel.
        exec_time = max(ideal_compute_time, ideal_memory_time)

        thrpt = num_ops/exec_time if exec_time else 0
        com_to_mem_ratio = ideal_compute_time/ideal_memory_time if ideal_memory_time else 0
        boundedness = 'C' if com_to_mem_ratio > 1 else 'M'



        ret = {
            'Op Type': self.get_op_type(),
            'Dimension': self.dim[:self.get_effective_dim_len()],
            'Bound': boundedness,
            'C/M ratio': com_to_mem_ratio,
            'Op Intensity': op_intensity,
            f'Latency ({unit.unit_time})': unit.raw_to_unit(exec_time, type='T'),
            f'Cycles': exec_time*system.frequency,
            f'Num ops ({unit.unit_flop})': unit.raw_to_unit(num_ops, type='O'),
            f'Input_a ({unit.unit_mem})': unit.raw_to_unit(input_a_size, type='M'),
            f'Input_w ({unit.unit_mem})': unit.raw_to_unit(input_w_size, type='M'),
            f'Output ({unit.unit_mem})': unit.raw_to_unit(output_size, type='M'),
            f'Total Data ({unit.unit_mem})': unit.raw_to_unit(num_data, type='M'),
            f'Throughput ({unit.unit_compute})': unit.raw_to_unit(thrpt, type='C'),
            f'Compute Cycles': ideal_compute_time*system.frequency,
            f'Memory Cycles': ideal_memory_time*system.frequency,

        }

        return ret










