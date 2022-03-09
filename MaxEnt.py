from platform import node
from xml.sax.handler import all_features
from matplotlib.pyplot import get
import phirl.api as ph
import pandas as pd
import numpy as np  
import itertools

data = pd.read_csv('tree_df.csv',delimiter=',')
mutation = pd.read_csv('mhn.csv', delimiter=',')

def get_trees():
    Forest_naming = ph.ForestNaming()
    trees = ph.parse_forest(data,Forest_naming)
    return trees

def get_all_trajectory_states_features(trees):
    all_trajectory = []
    all_states = []
    state_combinations = [list(i) for i in itertools.product([0, 1], repeat=5)]
    all_features = []

    for i in range(1,len(trees)+1):
        # trajectory
        trajectory = ph.list_all_trajectories(trees[i],max_length=10)
        all_trajectory.append(trajectory)

        # state and features
        state = [0,0,0,0,0]
        features = [0]*32

        for j in range(len(trajectory[0])):
            
            if trajectory[0][j].mutation > 0:
                state[trajectory[0][j].mutation-1] = 1
                idx = state_combinations.index(state)
                features[idx] = 1
            
        all_states.append(state)
        all_features.append(features)

    return all_trajectory, all_features, all_states

#def initial_reward()
#reward = np.zeros(5)
# all states have the same weights of reward
#reward[:] = 1


#Not sure about this
def feature_expectation_from_trajectories(all_features, all_trajectory):
    feature_expectation = [0]*32

    for i in range(len(all_features)):
        feature_expectation = [a + b for a, b in zip(feature_expectation, all_features[i])]
    
    feature_expectation = sum(feature_expectation)/len(all_trajectory)

    return feature_expectation

if __name__=='__main__':
    trees = get_trees()
    all_trajectory, all_features, all_states = get_all_trajectory_states_features(trees)
    feature_expectation = feature_expectation_from_trajectories(all_features, all_trajectory)
    
    
    
    


