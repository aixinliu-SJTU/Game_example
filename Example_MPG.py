#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from game_theory import GameEnvironment
import make_network_n
from structure_mat import get_t1_matrix, cal_KR_seq


# ==================== network ========================

reward_matrix_0 = np.array(
                [[-1,-10],
                 [0,-8]]
                )


reward_matrix_1 = np.array(
                [[-1,-10],
                 [0,-8]]
                )

reward_matrix_2 = np.array(
                [[-1,-10],
                 [0,-8]]
                )


reward_matrix_3 = np.array(
                [[-1,-10],
                 [0,-8]]
                )

reward_matrix_4 = np.array(
                [[-1,-10],
                 [0,-8]]
                )


reward_matrix_5 = np.array(
                [[-1,-10],
                 [0,-8]]
                )

reward_matrix_6 = np.array(
                [[-1,-10],
                 [0,-8]]
                )


reward_matrix_7 = np.array(
                [[-1,-10],
                 [0,-8]]
                )

reward_matrix_8 = np.array(
                [[-1,1],
                 [1,-1]]
                )


reward_matrix_9 = np.array(
                [[1,-1],
                 [-1,1]]
                )


reward_mats = [reward_matrix_0, reward_matrix_1, reward_matrix_2, reward_matrix_3, reward_matrix_4,
              reward_matrix_5, reward_matrix_6, reward_matrix_7, reward_matrix_8, reward_matrix_9]


# ===================== network =========================

network = make_network_n.create_10_net()

# ============== conflicting analysis ===================

strategy_len = reward_matrix_1.shape[-1]
num_agents = network.number_of_nodes()

max_num = strategy_len ** num_agents

assert strategy_len == 2
max_decode_length = len(bin(max_num-1)[2:]) 

env = GameEnvironment(network, strategy_len, reward_mats)

z = get_t1_matrix(env, max_num, max_decode_length)

seq, _ = cal_KR_seq(z, strategy_len)

num_agents = 10
max_strategy = 1024  

payoff_matrix = np.zeros((num_agents, max_strategy))

for strategy_profile in range(1, max_strategy + 1):
   
    strategy_list = [int(x) for x in format(strategy_profile - 1, '010b')]
   
    env.reset(strategy_list)

    env.step()

    payoff_vector = env.get_payoff_vector(strategy_list)

    for player_index, payoff_value in enumerate(payoff_vector):
        payoff_matrix[player_index, strategy_profile - 1] = payoff_value

total_mat = []

for player_index, payoff_vector in enumerate(payoff_matrix):
    print(f"Player {player_index} Payoff Matrix: {payoff_vector}")
    total_mat.append(payoff_vector)
total_mat = np.concatenate(total_mat)
np.save('reward_vec', total_mat)

cost_mat = np.stack(payoff_matrix, axis=0)

loaded_data = np.load('EE_player10_mat.npz', allow_pickle=True)
EE_total = loaded_data['EE_total']
segment_table = loaded_data['segment_table']
reward_vec = np.load('reward_vec.npy')

A = EE_total
b = reward_vec

if np.linalg.matrix_rank(A) == min(A.shape):
    x = np.linalg.solve(A, b) # The system has a unique solution
    np.save('solver_x.npy', x)  
else:
    x = np.array([])  

if x.size > 0:
    np.save('solver_x.npy', x)

start_idx = 6144
end_idx = 10240
subset_x = x[start_idx:end_idx]

epsilon = 1e-11
matching_indices = np.where((subset_x < -epsilon) | (subset_x > epsilon))[0]

if matching_indices.size > 0:
    matching_indices = matching_indices + start_idx + 1  
    print("The position of the element that satisfies the condition is:", matching_indices)
else:
    print("There are no matching elements.")

num_blocks = 10

block_size = 10240 // num_blocks

for column_i in matching_indices-1:
    column_vector = EE_total[:, column_i]
    block_positions_with_minus_or_one = set()  
    
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size
        
        block_vector = column_vector[start_idx:end_idx]
        
        if np.any(block_vector == -1) or np.any(block_vector == 1):
            block_positions_with_minus_or_one.add(i)
    
unique_block_combinations = set()

for column_i in matching_indices - 1:
    column_vector = EE_total[:, column_i]
    block_positions_with_minus_or_one = set()  
    
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size
        
        block_vector = column_vector[start_idx:end_idx]
        
        if np.any(block_vector == -1) or np.any(block_vector == 1):
            block_positions_with_minus_or_one.add(i+1) #Change the count from 0 to 1

    
    unique_block_combinations.add(frozenset(block_positions_with_minus_or_one))


for combination in unique_block_combinations:
    print(f"Players with a conflict of interest: {set(combination)}")


