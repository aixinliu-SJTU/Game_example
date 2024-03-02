# -*- coding: utf-8 -*-
import numpy as np

class GameEnvironment:
    def __init__(self, network, dim_strategy, reward_mats):
        self.G = network
        self.num_agent = self.G.number_of_nodes()
        assert isinstance(dim_strategy, int)
        self.dim_strategy = dim_strategy
        self.reward_mats = reward_mats  # 列表包含每个节点的reward_mat
        
    def reset(self, strategy):
        assert len(strategy) == self.num_agent
        self.strategy = strategy
        
    def step(self):
        current_strategy = self.get_current_strategy()
        self.do_strategy(current_strategy)
        self.calculate_reward()
        self.update_strategy()
        
    def get_current_strategy(self):
        return self.strategy
        
    def do_strategy(self, current_strategy):
        self.set_agent_strategy(current_strategy)
        
    def calculate_reward(self):
        iter_reward = []
        for node_i_index, node_i_adj in self.G.adjacency():
            cumul_reward_of_node_i = 0
            for node_k_index in node_i_adj:
                node_k_reward_mat = self.reward_mats[node_i_index]  # 使用节点特定的reward_mat
                reward_k = self._get_reward_from_interact(node_i_index, node_k_index, node_k_reward_mat)
                cumul_reward_of_node_i += reward_k
            iter_reward.append(cumul_reward_of_node_i)
            
    def update_strategy(self):
        new_strategy = self.MBRA()
        self.strategy = new_strategy
                
    def set_agent_strategy(self, current_strategy):
        for i in range(self.num_agent):
            self.G.nodes[i]['strategy'] = int(current_strategy[i])
        
    def _get_reward_from_interact(self, node_i_index, node_j_index, mat=None):
        reward_mat = mat
        node_i_strategy = self.G.nodes[node_i_index]['strategy']
        node_j_strategy = self.G.nodes[node_j_index]['strategy']
        reward = reward_mat[node_i_strategy][node_j_strategy]
        return reward
    
    def MBRA(self):
        new_strategy = []
        for node_i_index, node_i_adj in self.G.adjacency():
            act_list_cumul_reward = []
            for act in range(self.dim_strategy):
                act_reward_assume = 0
                for node_k_index in node_i_adj:
                    node_adj_strategy = self.G.nodes[node_k_index]['strategy']
                    reward_mat_k = self.reward_mats[node_i_index]  # 使用节点特定的reward_mat
                    reward_k = reward_mat_k[act][node_adj_strategy]
                    act_reward_assume += reward_k
                act_list_cumul_reward.append(act_reward_assume)
            node_i_new_strategy = np.argmax(act_list_cumul_reward)
            new_strategy.append(node_i_new_strategy)
        return new_strategy
    
    def get_payoff_vector(self, strategy):
        assert len(strategy) == self.num_agent
        payoff_vector = []
        for node_i_index, node_i_adj in self.G.adjacency():
            cumul_reward_of_node_i = 0
            for node_k_index in node_i_adj:
                node_k_reward_mat = self.reward_mats[node_i_index]  # 使用节点特定的reward_mat
                reward_k = self._get_reward_from_interact(node_i_index, node_k_index, node_k_reward_mat)
                cumul_reward_of_node_i += reward_k
            payoff_vector.append(cumul_reward_of_node_i)
        return payoff_vector
