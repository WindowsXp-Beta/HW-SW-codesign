import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from unit import Unit


def plot_roofline_background(system, max_x,unit):
    op_intensity = system.flops/system.offchip_mem_bw
    flops = unit.raw_to_unit(system.op_per_sec, type='C')
    max_x = max_x
    turning_points = [[0, 0], [op_intensity, flops], [max(max_x, 1.5*op_intensity), flops]]
    turning_points = np.array(turning_points)
    plt.plot(turning_points[:,0], turning_points[:,1], c='grey')

    plt.xlabel('Op Intensity (FLOPs/Byte)')
    plt.ylabel(f'{unit.unit_compute.upper()}')


def dot_roofline(df,system):
    unit = Unit()
    max_x = max(df['Op Intensity'])
    plot_roofline_background(system, max_x,unit)
    markers = ["." , "," , "o" , "v" , "^" , "<", ">"]
 
    for i in range(len(df)):
        op_intensity = df.loc[i, 'Op Intensity']
        thrpt = df.loc[i, 'Throughput (Tflops)']
        plt.scatter(op_intensity, thrpt, label=f'{df.loc[i, "Op Type"]}-{i}', marker=markers[i%7])
    plt.legend(bbox_to_anchor =(1, 0.9), ncol = 1)
