# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:48:19 2024

@author: VERMA Anshuman
Ph.D. Student

Scanning Probe Microscopy Group (Renner Group)

Department of Quantum Matter Physics
Université de Genève
24 Quai Ernest-Ansermet, Genève-1205, Suisse

Module Name: GenIvAnPy
File Name: Refined_File_Reader.py

This version also creates a "try" to create fourier transform maps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime


import tkinter as tk
from tkinter import filedialog
import pickle






### HEADER STRING EXTRACTOR
def get_header_bs(file_path):
    header_string_file = open(file_path,"rb")
    long_header_string = header_string_file.read()

    header_string = long_header_string[0:5000] #Just for Testing

    header_end_finder = long_header_string.find(b"HEADER_END")

    byte_shift = header_end_finder + len("HEADER_END:\r\n")

    unarranged_header = long_header_string[:header_end_finder + len("HEADER_END:\r\n")].decode("utf-8")
    arranged_header= unarranged_header.split("\r\n")[:-2] #HEADER_END is not present in this string array

    # Permanent Variable
    header = arranged_header

    return(header,byte_shift)





def get_long_file(file_path, byte_shift):
    long_file = np.fromfile(file_path,dtype = ">f4",offset = byte_shift)
    return (long_file)



#Functions...
def contains(variable = "not found",array = []):
    #header will be passed as the array in most cases!
    index_array = []
    final_array = []
    for index,element in enumerate(array):
        if variable.lower() in element.lower():
            # print("Found!")
            index_array.append(index)
            final_array.append(element)
            # print(index)

    if (index_array == [] and final_array == []):
        print("!!!INVALID!!!")
    else: 
        return(index_array,final_array)


# Finds the Number of Parameters
def number_of_parameters(file_path):
    header,byte_shift = get_header_bs(file_path) #getting the header and byte shift
    index, nop = contains("Experiment Parameters", header)
    nop_string = nop[0]
    equals_finder = nop_string.find("=")
    nop1 = nop_string[equals_finder+1:]
    nop2 = nop1.replace('"','')
    number_of_parameters = len(nop2.split(";"))
    return(number_of_parameters)

def number_of_parameters2(file_path):
    header,byte_shift = get_header_bs(file_path)
    index, nop = contains("Experiment Parameters", header)
    nop_string = nop[0]
    equals_finder = nop_string.find("=")
    nop1 = nop_string[equals_finder+1:]
    nop2 = nop1.replace('"','')
    number_of_parameters = len(nop2.split(";"))
    return(number_of_parameters,nop_string,nop)

def parameters(file_path):
    header,byte_shift = get_header_bs(file_path)
    index, p = contains("Experiment parameters", header)
    p_string = p[0]
    equals_finder = p_string.find("=")
    p1 = (p_string[equals_finder+1:])
    p2 = p1.replace('"','') 
    parameters = p2.split(";")
    return(parameters)


def number_of_fixed_parameters(file_path):
    header,byte_shift = get_header_bs(file_path)
    index, nofp = contains("Parameters", header)
    nofp_string = nofp[0]
    equals_index = nofp_string.find("=")
    nofp_1 = (nofp_string[equals_index+1:]) #Actual No. of Parameters in an integer format...
    nofp_2 = nofp_1.replace('"','')
    count = 0 #For Counting the Number of Parameters
    for letter in nofp_2:
        if(letter==";"):
            count = count+1
    count = count+1
    return(count)



def fixed_parameters(file_path):
    header,byte_shift = get_header_bs(file_path)
    index, fp = contains("Fixed parameters", header)
    fp_string = fp[0]
    equals_index = fp_string.find("=")
    fp_1 = (fp_string[equals_index+1:]) 
    fp_2 = fp_1.replace('"','')
    fixed_parameters = fp_2.split(";")
    return(fixed_parameters)



def total_number_of_parameters(file_path):
    total_number_of_parameters = number_of_parameters(file_path) + number_of_fixed_parameters(file_path)
    return(total_number_of_parameters)


def all_parameters(file_path):
    all_parameters = fixed_parameters(file_path) + parameters(file_path)
    return(all_parameters)


def grid_dimensions(file_path):
    header,byte_shift = get_header_bs(file_path)
    index,dim = contains("Grid dim", header)
    dim_string = dim[0]
    dim1 = dim_string.replace('"','')
    dim2 = dim1.replace(" ","")
    equals_index = dim2.find("=")
    x_index = dim2.find("x")
    dim_x = dim2[equals_index+1:x_index]
    dim_y = dim2[x_index+1:]
    dim_x,dim_y = int(dim_x),int(dim_y)
    return(dim_x,dim_y)

def points(file_path):
    header,byte_shift = get_header_bs(file_path)
    index,p = contains("Points", header)
    points = int((p[0].split("="))[1])
    # print(p)
    return(points)
    

def channels(file_path):
    header,byte_shift = get_header_bs(file_path)
    index,c = contains("Channels", header)
    c1 = c[0]
    equals_index = c1.find("=")
    c2 = c1[equals_index+1:]
    c3 = c2.replace('"',"")
    channels = c3.split(";")
    return(channels)

def number_of_channels(file_path):
    number_of_channels = len(channels(file_path))
    return(number_of_channels)


###############
###############

#Just for Reference
def byte_calculator(file_path,byte_shift):
    long_file = get_long_file(file_path, byte_shift)
    total_bytes = (grid_dimensions(file_path)[0] * grid_dimensions(file_path)[1]) * ((number_of_fixed_parameters(file_path) + number_of_parameters(file_path)) + (points(file_path) * number_of_channels(file_path)))
    if (total_bytes==len(long_file)):
        print("Yes, the total number of bytes match with the Formula!")
    return(total_bytes)



###############
def partitions(file_path):
    tnop = total_number_of_parameters(file_path)
    p = points(file_path)
    noc = number_of_channels(file_path)
    data_block_size = tnop+p*noc
    return(data_block_size)


# def long_data_blocks(file_path, byte_shift): #RETURNS THE ENTIRE FILE
#     data_block_size = partitions(file_path)
#     long_file = get_long_file(file_path, byte_shift)
#     grid_size = grid_dimensions(file_path)[0]*grid_dimensions(file_path)[1]
#     long_file_data_blocks = long_file.reshape(grid_size,data_block_size) 
#     return(long_file_data_blocks)

def long_data_blocks(file_path, byte_shift):
    data_block_size = partitions(file_path)  # e.g., 2100 or 8419
    long_file = get_long_file(file_path, byte_shift)  # 1D array
    total_points = long_file.size

    if total_points % data_block_size != 0:
        raise ValueError(f"Cannot reshape array of size {total_points} into blocks of size {data_block_size}")

    num_blocks = total_points // data_block_size
    long_file_data_blocks = long_file.reshape((num_blocks, data_block_size))
    
    return long_file_data_blocks

def data_matrix(file_path,byte_shift):
    long_file = get_long_file(file_path, byte_shift)
    data_block_size = partitions(file_path)
    grid_size = grid_dimensions(file_path)[0]*grid_dimensions(file_path)[1]

    long_file_data_blocks = long_file.reshape(grid_dimensions(file_path)[0],grid_dimensions(file_path)[1],data_block_size) 
    return(long_file_data_blocks)


def parameter_dictionary(file_path,byte_shift):
    grid_x = grid_dimensions(file_path)[0]
    grid_y = grid_dimensions(file_path)[1]
    all_params = all_parameters(file_path)
    nop = len(all_params)
    data = data_matrix(file_path,byte_shift)
    
    parameter_dictionary = {param: np.zeros((grid_x,grid_y)) for param in all_params}

    for index,parameter in enumerate(parameter_dictionary.keys()):
        for j in range(0, grid_x):
            for k in range(0, grid_y):
        
                parameter_dictionary[parameter][j][k] = data[j][k][index]
                
    return(parameter_dictionary)

def bias_sweep_array(file_path, byte_shift):
    parameter_dict = parameter_dictionary(file_path, byte_shift)
    ss = parameter_dict['Sweep Start']
    se = parameter_dict['Sweep End']
    pts1 = points(file_path)
    
    if np.all(ss == ss[0,0]) and np.all(se == se[0,0]):
        # print(f"The Bias Sweep Start and Sweep End for all {grid_dimensions()[0]} x {grid_dimensions()[1]} points in the grid are the same!")
        sweep_start = ss[0,0]
        sweep_end = se[0,0]
        Bias_Array = np.linspace(sweep_start,sweep_end,pts1)
        return(Bias_Array)
               
        
    else:
        print("ERROR! THE BIAS SWEEP AND SWEEP END FOR ALL GRID POINTS ARE NOT THE SAME!")
        ## MAYBE DESIGN ANOTHER FUNCTION???
        return()
    


def channel_dictionary(file_path,byte_shift):
    raw_data = data_matrix(file_path,byte_shift)
    g_x = grid_dimensions(file_path)[0]
    g_y = grid_dimensions(file_path)[1]
    parts = partitions(file_path)
    pts = points(file_path)
    tonp = total_number_of_parameters(file_path) #total number of parameters (fixed parameters + parameters)
    channel_dictionary = {channel: np.zeros(((g_x,g_y,pts))) for channel in channels(file_path)}
    
    for i in range(g_x):
        for j in range(g_y):
            for index,channel in enumerate(channel_dictionary.keys()):
                channel_dictionary[channel][i,j] = raw_data[i,j][tonp+pts*index:tonp+(index+1)*pts]
    return(channel_dictionary)




#Generates a dataframe which is flattened out and needs to be rechanged to (grid_x*grid_y*points())
def df_generator(file_path,byte_shift):
    cd = channel_dictionary(file_path,byte_shift)
    keys = cd.keys()
    bias = bias_sweep_array(file_path, byte_shift)
    df = pd.DataFrame()
    
    for key in keys:
        df[key] = np.array((cd[key].flatten()).tolist())
    # df['Bias (V)'] = bias
    if len(bias) < len(df):
        bias = np.pad(bias, (0, len(df) - len(bias)), mode='constant', constant_values=np.nan)
    df['Bias (V)'] = bias
    return(df)
          


if __name__=="__main__":
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title = "Select the Nanonis Binary File")
    file_path = file_path.replace("/","\\") 
    
    #Calling the Functions!!!!!!
    header, byte_shift = get_header_bs(file_path)
    long_file_data = get_long_file(file_path, byte_shift)
    
    num_params = number_of_parameters(file_path)
    param_list = parameters(file_path)
    num_fixed_params = number_of_fixed_parameters(file_path)
    fixed_param_list = fixed_parameters(file_path)
    total_params = total_number_of_parameters(file_path)
    all_param_list = all_parameters(file_path)
    grid_dims = grid_dimensions(file_path)
    num_points = points(file_path)
    channel_list = channels(file_path)
    num_channels = number_of_channels(file_path)
    partition_size = partitions(file_path)
    long_blocks = long_data_blocks(file_path, byte_shift)
    bias_array = bias_sweep_array(file_path, byte_shift)
    
    df = df_generator(file_path,byte_shift)
    
    
    cd = channel_dictionary(file_path,byte_shift)
    
    # Saving the filepath as a pickle file
    with open("file_path.pkl", "wb") as f:
        pickle.dump(file_path,f)
    
    with open("byte_shift.pkl","wb") as f:
        pickle.dump(byte_shift,f)
        
    with open("header.pkl","wb") as f:
        pickle.dump(header,f)
    
    
    
    #Input Directory for Images

    #Datetime business
    dt = datetime.datetime.now()
    dt_array = [dt.year,dt.month,dt.day,dt.day,dt.hour,dt.minute,dt.second]
    f_dt = ''.join(str(i99) for i99 in dt_array)

    # dIdV_Maps(file_path = file_path,byte_shift = byte_shift)
    
    # #Single dIdV_Map
    # dIdV_Map()

    # df_to_SQL()
    # cd_from_SQL = SQL_to_df()
    
        
##################  
   
