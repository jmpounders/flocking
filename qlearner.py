import numpy as np
import pickle
from datetime import datetime

from flock import Flock

class QLearner(object):
    def __init__(self, alpha, gamma, n_states, n_actions, n_agents, eps, tol=1e-8):
        self.alpha = alpha
        self.gamma = gamma
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.eps = eps
        self.tol = tol

        Q_init = 0.0       
        self.Q_mats = [np.zeros((self.n_states, self.n_actions))+Q_init for _ in range(self.n_agents)]

        self.cost_episodes = None

    def compute_avg_q_matrix(self):
        Q_avg = np.zeros((self.n_states, self.n_actions))
        for Q in self.Q_mats:
            Q_avg += Q
        Q_avg /= len(self.Q_mats)

        return Q_avg

    def save(self, fname='qlearner'):
        np.save(f'{fname}.npy', self.compute_avg_q_matrix())
        with open(f'{fname}.pickle', 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fname='qlearner'):
        Q = np.load(f'{fname}.npy')
        with open(f'{fname}.pickle', 'rb') as f:
            qlearner = pickle.load(f)

        return qlearner, Q

    def run_episode(self, flock, episode_len, save_trajectories=False, learn=True):
        
        cost_avg = []
        if save_trajectories:
            trajectories_x = np.zeros((episode_len, flock.n_birds))
            trajectories_y = np.zeros((episode_len, flock.n_birds))
            directions_x = np.zeros((episode_len, flock.n_birds))
            directions_y = np.zeros((episode_len, flock.n_birds))

        state, cohesion, alignment, n_neighbors = flock.get_state()

        # random policy
        actions = np.random.choice(flock.turns.shape[0], size=flock.n_birds, replace=True)

        for ti in range(episode_len):

            flock.update(flock.turns[actions])
            if save_trajectories:
                trajectories_x[ti,:] = flock.position[:,0]
                trajectories_y[ti,:] = flock.position[:,1]
                directions_x[ti,:] = flock.direction[:,0]
                directions_y[ti,:] = flock.direction[:,1]

            state_new, cohesion_new, alignment_new, n_neighbors_new = flock.get_state()
            
            reward = flock.get_reward(cohesion_new, n_neighbors_new)

            cost_avg.append(-reward.mean())

            if learn:
                for i in range(flock.n_birds):
                    next_action_i = np.argmax(self.Q_mats[i][state_new[i], :])
                    exp_val = self.Q_mats[i][state_new[i], next_action_i]
                    self.Q_mats[i][state[i], actions[i]] = self.Q_mats[i][state[i], actions[i]] + self.alpha*(reward[i] + self.gamma*exp_val - self.Q_mats[i][state[i], actions[i]])
                    prob = np.random.uniform()
                    actions[i] = next_action_i if prob<1-self.eps else np.random.choice(self.n_actions)
            else:
                for i in range(flock.n_birds):
                    next_action_i = np.argmax(self.Q_mats[i][state_new[i], :])
                    actions[i] = next_action_i

            state, cohesion_new, n_neighbors = state_new, cohesion, n_neighbors_new

        output = [flock, np.mean(cost_avg)]
        if save_trajectories:
            output += [trajectories_x, trajectories_y, directions_x, directions_y]
            
        return output
            
    def run_episodes(self, n_episodes, episode_len, flock_params, save=True, cost_freq=10, eval_freq=50):
        self.cost_episodes = []

        for iter in range(n_episodes):
            flock_params['random_seed'] += 1
            flock = Flock(**flock_params)
            output = self.run_episode(flock, episode_len)
            cost_avg = output[1]
            self.cost_episodes.append(cost_avg)

            Q = self.compute_avg_q_matrix()
            self.Q_mats = [Q for _ in range(self.n_agents)]

            if iter%cost_freq==0:
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f'LERN {now} {iter}/{n_episodes}; Avg cost = {cost_avg:.4f}')
                self.save()

            if (iter%eval_freq==0) or (iter==n_episodes-1):
                self.eval(flock_params)

            if save:
                self.save()
        
        return self.cost_episodes

    def eval(self, flock_params):
        flock = Flock(**flock_params)
        output= self.run_episode(flock, 500, save_trajectories=True, learn=False)
        
        cost_avg = output[1]
        dir_x, dir_y = output[4], output[5]

        vdir = np.array((dir_x.sum(axis=1), dir_y.sum(axis=1)))
        polar_order = np.linalg.norm(vdir, axis=0)/dir_x.shape[1]

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('-------------------------')
        print(f'EVAL {now}; Avg cost = {cost_avg:.4f}; PO = {polar_order.mean():.4f}')
        print('-------------------------') 