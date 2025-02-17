
import operators as operators
import pandas as pd
import numpy as np



def analysis_model(model_operators, system, data_format='fp32'):
    roofline_list = []
    for i,operator_instance in enumerate(model_operators):
        roofline = operator_instance.get_roofline(system=system, data_format=data_format)
        if i==0:
            column = roofline.keys()
        roofline_list.append([roofline[c] for c in column])

    df = pd.DataFrame(np.array(roofline_list,dtype=object), columns=column, dtype=object)


    return df


#     return df
# def analysis_model(model_operators, system, tiling, Tm, Tk, Tn):
#     roofline_list = []
#     for i,operator_instance in enumerate(model_operators):
#         roofline = operator_instance.get_roofline(system=system, tiling=tiling, Tm=Tm, Tk=Tk, Tn=Tn)
#         if i==0:
#             column = roofline.keys()
#         roofline_list.append([roofline[c] for c in column])

#     df = pd.DataFrame(np.array(roofline_list,dtype=object), columns=column, dtype=object)


#     return df


# def analysis_model(model_operators, system, fusion: list[tuple] = []):
#     roofline_list = []
#     fusion_it = 0
#     for i,operator_instance in enumerate(model_operators):
#         fusion_pos = -2 # -2: no fusion, -1: last op, 0: first op, 1: mid op
#         if fusion_it < len(fusion):
#             if i == fusion[fusion_it][0]:
#                 fusion_pos = 0
#             elif i == fusion[fusion_it][-1]:
#                 fusion_pos = -1
#                 fusion_it += 1
#             elif i in fusion[fusion_it]:
#                 fusion_pos = 1
#         roofline = operator_instance.get_roofline(system=system, fusion_pos=fusion_pos)
#         if i==0:
#             column = roofline.keys()
#         roofline_list.append([roofline[c] for c in column])

#     df = pd.DataFrame(np.array(roofline_list,dtype=object), columns=column, dtype=object)


#     return df