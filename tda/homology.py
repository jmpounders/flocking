import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import euclidean_distances

from scipy.stats import gaussian_kde as kde

import gudhi as gd

from .dtm_filtration import dtm, weighted_rips_filtration

def bottleneck_distance(input1, input2):
    diagram1 = np.vstack(input1.homology)
    diagram2 = np.vstack(input2.homology)
    return gd.bottleneck_distance(diagram1, diagram2)

def _get_landscape_component(t_birth, t_death):
    """Get a component function for a feature that was born at t_birth and died at t_death.

    These are triangular functions, so only the three vertex points are returned.  The
    function is defined to be zero outside the 'tent' and can be linearly extrapolated within."""
    t_mid = (t_birth+t_death)/2
    landscape = [[t_birth, 0.0],
                 [t_mid,   t_mid-t_birth],
                 [t_death, 0.0]]
    return np.array(landscape)

class HomologyAnalysis(object):

    def __init__(self, complex_type, max_dimension=1, min_persistence=0.01):
        complex_types = {
            'rips':gd.RipsComplex,
        }

        assert complex_type in complex_types

        self.complex_type = complex_type
        self.make_complex = complex_types[complex_type]
        self.complex = None
        self.simplex_tree = None

        self.data = None
        self.max_dim = max_dimension
        self.min_persistence = min_persistence
        self.homology = None

        self.has_data = False
        self.has_complex = False
        self.has_persistence = False

    def generate_complex(self, points, complex_args=None):
        self.data = points
        self.has_data = True

        complex_args = complex_args if complex_args else {}
        self.complex = self.make_complex(points=points, **complex_args)
        self.simplex_tree = self.complex.create_simplex_tree(max_dimension=self.max_dim+1)
        self.has_complex = True

    def generate_complex_from_distances(self, D, complex_args=None):
        self.D = D
        self.has_data = False

        complex_args = complex_args if complex_args else {}
        self.complex = self.make_complex(distance_matrix=D, **complex_args)
        self.simplex_tree = self.complex.create_simplex_tree(max_dimension=self.max_dim+1)
        self.has_complex = True

    def generate_complex_from_dtm(self, points, m, p, filtration_max=np.inf):
        self.dtm_values = dtm(points, points, m)
        self.simplex_tree = weighted_rips_filtration(points, self.dtm_values, p, self.max_dim+1, filtration_max)
        self.has_complex = True

    def compute_persistence(self):
        assert self.has_complex

        self.homology = self.simplex_tree.persistence(homology_coeff_field=2,
                                                      persistence_dim_max=False,
                                                      min_persistence=self.min_persistence)

        intervals_in_dim = self.simplex_tree.persistence_intervals_in_dimension
        persistence_ints = [intervals_in_dim(d) for d in range(self.max_dim+1)]
        empty_dims = []
        for dim,persistence_int in enumerate(persistence_ints):
            if len(persistence_int)==0:
                print(f'Warning: dimension {dim} has no persistence intervals.')
                empty_dims.append(dim)
        for dim in empty_dims:
            persistence_ints[dim] = np.array([(0.0,0.0)])

        self.min_birth = min([min(persistence_int[:,0]) for persistence_int in persistence_ints if len(persistence_int)>0])
        self.max_death = 0
        for persistence_int in persistence_ints:
            deaths = persistence_int[:,1]
            self.max_death = max(self.max_death, np.max(deaths[np.isfinite(deaths)]))
        self.inf_val = self.max_death + (self.max_death - self.min_birth) * 0.1

        h = [pd.DataFrame(persistence_ints[dim], columns=['birth', 'death']) for dim in range(self.max_dim+1)]
        self.homology = [h_dim.replace(np.inf, self.inf_val) for h_dim in h]
        self.has_persistence = True


    def plot_persistence_diagram(self, ax=None, alpha=0.8):
        assert self.has_persistence

        ax = ax if ax else plt.gca()

        ax.plot((self.min_birth,self.inf_val), (self.min_birth,self.inf_val), linewidth=1, color='grey')
        ax.plot((self.min_birth,self.inf_val), (self.inf_val,self.inf_val), linewidth=1, color='grey')
        for dim in range(self.max_dim+1):
            ax.scatter(self.homology[dim]['birth'], self.homology[dim]['death'], label=f'H{dim}', alpha=alpha)

        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='lower right')

        return ax

    def plot_persistence_barcode(self, ax=None):
        spans = [self.homology[dim]['death']-self.homology[dim]['birth'] for dim in [0,1]]

        offset = 0
        ind = []
        for dim in range(self.max_dim+1):
            ind.append(self.homology[dim].index.values + offset)
            offset += len(ind[-1])


        ax = ax if ax else plt.gca()
        colors = ['blue', 'red']*10
        for dim in range(self.max_dim+1):
            ax.barh(ind[dim], left=self.homology[dim]['birth'], width=spans[dim], color=colors[dim])

        return ax

    def plot_persistence_landscape(self, dim, ax=None):
        ax = ax if ax else plt.gca()
        for interval in self.homology[dim].values:
            landscape = _get_landscape_component(*interval)
            ax.plot(landscape[:,0], landscape[:,1])

        return ax

    def plot_betti_curve(self, dim, min_val=None, max_val=None, num_points=100, ax=None):
        min_val = min_val if min_val else self.min_persistence
        max_val = max_val if max_val else self.max_death
        fval = np.linspace(min_val, max_val, num_points)
        betti_nums = np.zeros(len(fval))
        for i,val in enumerate(fval):
            mask = (self.homology[dim]['birth']<=val) & (self.homology[dim]['death']>=val)
            betti_nums[i] = self.homology[dim][mask].shape[0]

        ax = ax if ax else plt.gca()
        ax.plot(fval, betti_nums)

        return ax

