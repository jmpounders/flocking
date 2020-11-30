import numpy as np
import scipy.spatial.distance as dist

from sklearn.neighbors import KDTree

# TODO
# - Vary alpha in epsilon in the qlearner to encourage more exploration early
#
# Notes
# - The state should at least include alignment.
# - The state may also include a neighbor density function
# - Without the neighbor density function, the flock tends
#   to oscillate more;  they appear to not have a memory
#   of density.  IOW there is not positively-rewarded state
#   of "nice density" so they are constantly searching for it.

class Flock(object):
    """Object representing a flock of birds in two dimensions."""
    
    num_turns = 8
    num_bins_alignment = 8
    num_bins_neighbors = 4
    
    def __init__(self, n_birds, speed, time_step, 
                 bounding_box=None,
                 perceptual_radius=2,
                 birth_box=None,
                 random_seed=42):
        self.n_birds = n_birds
        self.speed = speed
        self.delta_t = time_step
        self.bbox = bounding_box
        self.birth_box = self.bbox if birth_box is None else birth_box
        self.perceptual_radius = perceptual_radius
        self.seed = random_seed
        
        self.turns = np.linspace(-np.pi/6, np.pi/6, self.num_turns)
        
        self.TOL = 1e-6
        
        self.position, self.direction = self.initialize()
        self.neighbors_in_range = None

        self.n_states = self.num_bins_alignment*self.num_bins_neighbors
        self.state_bin_boundaries_neigh = np.linspace(-1-self.TOL, 1+self.TOL, self.num_bins_neighbors)
        self.state_bin_boundaries_align = np.linspace(-1-self.TOL, 1+self.TOL, self.num_bins_alignment) 

        self.ideal_neighbors = 20

    def initialize(self):
        """Randomly initialize positions and directions of the flock."""
        np.random.seed(self.seed)
        
        position_x = np.random.uniform(self.birth_box[0], self.birth_box[1] ,self.n_birds)
        position_y = np.random.uniform(self.birth_box[0], self.birth_box[1] ,self.n_birds)
        direction_theta = np.random.uniform(0.0, 2*np.pi, self.n_birds)
        
        position = np.vstack((position_x, position_y)).T
        direction = np.vstack((np.cos(direction_theta), np.sin(direction_theta))).T
        
        return position, direction

    def get_neighbors(self):
        kdt = KDTree(self.position)
        neighbors, distances = kdt.query_radius(self.position, 
                                                r=self.perceptual_radius, 
                                                return_distance=True, 
                                                count_only=False)

        self.neighbors_in_range = []
        self.neighbor_distances_in_range = []
        for b in range(self.n_birds):
            nbrs, dists = neighbors[b], distances[b]

            rel_pos = self.position[nbrs,:]-self.position[b,:]
            in_perception = np.dot(rel_pos, self.direction[b,:])
            inds = np.where((in_perception>0)&(dists>0))[0]
            self.neighbors_in_range.append(nbrs[inds])
            self.neighbor_distances_in_range.append(dists[inds])

        return self.neighbors_in_range

    def compute_center_of_mass(self):
        com = np.zeros_like(self.position)
        for b in range(self.n_birds):
            nbr_inds = self.neighbors_in_range[b]
            if nbr_inds.shape[0]>0:
                com[b,:] = np.mean(self.position[nbr_inds,:], axis=0)
            else:
                com[b,:] = np.inf

        return com

    def compute_average_velocity(self):
        avg_vel = np.zeros_like(self.direction)
        for b in range(self.n_birds):
            nbr_inds = self.neighbors_in_range[b]
            if nbr_inds.shape[0]>0:
                avg_vel[b,:] = np.mean(self.direction[nbr_inds,:], axis=0)
            else:
                avg_vel[b,:] = np.inf

        return avg_vel
        
    
    def update(self, turn_angles=None, delta_v=None, apply_bcs=True):
        """Update the positions and directions of birds based on a set of turn angles."""
        if turn_angles is not None:
            direction_new_x = self.direction[:,0]*np.cos(turn_angles) - self.direction[:,1]*np.sin(turn_angles)
            direction_new_y = self.direction[:,0]*np.sin(turn_angles) + self.direction[:,1]*np.cos(turn_angles)
            self.direction = np.vstack((direction_new_x, direction_new_y)).T
        else:
            self.direction += delta_v
            norm = np.linalg.norm(self.direction, axis=1)
            for i in range(self.direction.shape[1]):
                self.direction[:,i] /= norm

        self.position += self.speed*self.direction*self.delta_t
        if apply_bcs:
            self._apply_boundary_conditions()
        
    def _apply_boundary_conditions(self):
        """Apply periodic boundary conditions on the flock."""
        boundary_conds = {
            'left':{
                'dim':0,
                'periodic_boundary':1,
                'dist_to_boundary':self.position[:,0]-self.bbox[0],
                'on_boundary':(self.position[:,0]-self.bbox[0])<self.TOL,
                'exiting':np.dot(self.direction, np.array([-1,0])) > self.TOL
            },
            'right':{
                'dim':0,
                'periodic_boundary':0,
                'dist_to_boundary':self.bbox[1]-self.position[:,0],
                'on_boundary':(self.bbox[1]-self.position[:,0])<self.TOL,
                'exiting':np.dot(self.direction, np.array([1,0])) > self.TOL
            },
            'up':{
                'dim':1,
                'periodic_boundary':0,
                'dist_to_boundary':self.bbox[1]-self.position[:,1],
                'on_boundary':(self.bbox[1]-self.position[:,1])<self.TOL,
                'exiting':np.dot(self.direction, np.array([0,1])) > self.TOL
            },
            'down':{
                'dim':1,
                'periodic_boundary':1,
                'dist_to_boundary':self.position[:,1]-self.bbox[0],
                'on_boundary':(self.position[:,1]-self.bbox[0])<self.TOL,
                'exiting':np.dot(self.direction, np.array([0,-1])) > self.TOL
            }
        }
        for _, condition in boundary_conds.items():
            idx = np.where(condition['on_boundary'] & condition['exiting'])[0]
            multi = 1 if condition['periodic_boundary']==1 else -1
            self.position[idx,condition['dim']] = self.bbox[condition['periodic_boundary']] + multi*condition['dist_to_boundary'][idx]
            
    def get_state(self):
        """Return the state of each bird in the flock.
        
        This is the state used by the qlearner."""
        self.get_neighbors()

        cohesion = np.zeros(self.n_birds)
        alignment = np.zeros(self.n_birds)
        n_neighbors = np.zeros(self.n_birds, dtype=int)

        for i in range(self.n_birds):
            nbrs = self.neighbors_in_range[i]
            n_neighbors[i] = nbrs.shape[0]
            if nbrs.shape[0]==0:
                cohesion[i] = 0.0
                continue

            avg_direction = self.direction[nbrs,:].mean(axis=0)
            avg_direction = avg_direction/np.linalg.norm(avg_direction)

            cohesion[i] = np.dot(avg_direction, self.direction[i,:])
            alignment[i] = np.cross(avg_direction, self.direction[i,:])

        state_n = 1/(1 + np.exp(self.ideal_neighbors-n_neighbors))
        state_n = np.digitize(state_n, self.state_bin_boundaries_neigh)-1
        state_a = np.digitize(alignment, self.state_bin_boundaries_align)-1
        state = state_a + self.num_bins_alignment*state_n

        assert np.all(state_n<self.num_bins_neighbors)
        assert np.all(state_a<self.num_bins_alignment)

        return state, cohesion, alignment, n_neighbors

    def get_reward(self, cohesion, n_neighbors):
        r_n = 1 - np.abs(self.ideal_neighbors-n_neighbors)/(n_neighbors+1)
        r_c = cohesion

        return (r_n+r_c)/2
        
    def plot(self, figsize=(5,5)):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        ax.quiver(self.position[:,0], self.position[:,1], self.direction[:,0], self.direction[:,1])
        ax.axis('equal')
        ax.set_xlim(self.bbox)