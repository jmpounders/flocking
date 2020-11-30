import numpy as np

class Boids(object):

    def __init__(self, flock, com_norm=10, dist_thresh=0.5, vel_norm=2):
        self.flock = flock
        self.com_norm = com_norm
        self.dist_thresh = dist_thresh
        self.vel_norm = vel_norm

    def rule_one(self):
        """Fly toward the neighborhood center of mass."""
        com = self.flock.compute_center_of_mass()
        delta_v = (com - self.flock.position)/self.com_norm
        return delta_v

    def rule_two(self):
        """Keep a small distance away from neighbors."""
        delta_v = np.zeros_like(self.flock.direction)
        for b in range(self.flock.n_birds):
            if self.flock.neighbor_distances_in_range[b].shape[0]==0:
                continue
            if np.all(self.flock.neighbor_distances_in_range[b]>=self.dist_thresh):
                continue
            nbrs = np.where(self.flock.neighbor_distances_in_range[b]<self.dist_thresh)[0]
            nbrs = self.flock.neighbors_in_range[b][nbrs]
            delta_v[b,:] = np.mean(self.flock.position[b,:] - self.flock.position[nbrs,:], axis=0)

        return delta_v

    def rule_three(self):
        """Try match neighbors velocity."""
        avg_vel = self.flock.compute_average_velocity()
        delta_v = (avg_vel - self.flock.direction)/self.vel_norm
        return delta_v

    def update(self):
        self.flock.get_neighbors()
        delta_v = self.rule_one()
        delta_v += self.rule_two()
        delta_v += self.rule_three()
        
        delta_v[delta_v==np.inf] = 0.0

        return delta_v

    def run_episode(self, episode_len, save_trajectories=False):
        
        cost_avg = []
        if save_trajectories:
            trajectories_x = np.zeros((episode_len, self.flock.n_birds))
            trajectories_y = np.zeros((episode_len, self.flock.n_birds))
            directions_x = np.zeros((episode_len, self.flock.n_birds))
            directions_y = np.zeros((episode_len, self.flock.n_birds))

        _, n_neighbors = self.flock.get_state()
        for ti in range(episode_len):
            delta_v = self.update()
            self.flock.update(delta_v=delta_v, apply_bcs=True)
            if save_trajectories:
                trajectories_x[ti,:] = self.flock.position[:,0]
                trajectories_y[ti,:] = self.flock.position[:,1]
                directions_x[ti,:] = self.flock.direction[:,0]
                directions_y[ti,:] = self.flock.direction[:,1]

            _, n_neighbors_new = self.flock.get_state()
            cost = (n_neighbors_new < n_neighbors).astype(int)
            cost_avg.append(cost.mean())

        output = [self.flock, np.mean(cost_avg)]
        if save_trajectories:
            output += [trajectories_x, trajectories_y, directions_x, directions_y]
            
        return output
