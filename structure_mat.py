import numpy as np
import scipy.linalg as sl

from utils import one_hot, oct2bin

def get_t1_strategy(env, init_strategy):
    iterations = 1
    
    env.reset(init_strategy)
    for i in range(iterations):
        env.step() 

    return env.strategy

def get_t1_matrix(env, max_num, max_decode_length):
    t1_mat = []
    for i in range(max_num):
        init_strategy_format = list(oct2bin(i, max_decode_length))
        t1_strategy = get_t1_strategy(env, init_strategy_format)
        
        t1_mat.append(t1_strategy)
    return np.array(t1_mat).T


def calculate_col_KR_pro(vec, strategy_len):
    assert len(vec.shape) == 1 and vec.shape[0] > 1
    
    kr_result = one_hot(vec[0], strategy_len)
    for value in vec[1:]:
        kr_result = sl.khatri_rao(kr_result, one_hot(value, strategy_len))
        
    assert kr_result.sum()==1
    kr_result = kr_result.reshape(-1)
    
    kr_index = int(np.where(kr_result == 1)[0])
    
    return kr_index, kr_result

def cal_KR_seq(struc_mat, strategy_len):
    kr_list = []
    kr_vec = []
    
    for i in range(struc_mat.shape[-1]):
        index, vec = calculate_col_KR_pro(struc_mat[:,i], strategy_len)
        
        kr_list.append(index)
        kr_vec.append(vec)
    return np.array(kr_list), np.array(kr_vec).T
    # print(np.array(kr_list)+1)


def test_net_iteration(env, init_strategy, iterations=1):
    # iterations = 1
    strategy_history = []
    env.reset(init_strategy)
    
    for i in range(iterations):
        if env.strategy not in strategy_history:
            strategy_history.append(env.strategy)
        
        env.step() 
    print('init strategy: %s | final: ' % init_strategy, strategy_history)