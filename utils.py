import numpy as np

def oct2bin(value, max_length):
    # max_length = 5
    
    bin_value = bin(value)[2:].rjust(max_length,'0')
    return bin_value

def one_hot(index, length):
    empty = np.zeros([length,1])
    empty[index] = 1
    return empty