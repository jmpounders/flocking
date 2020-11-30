import flock
import qlearner as ql

if __name__=='__main__':

    alpha = 0.005
    gamma = 0.0
    n_states = flock.Flock.num_bins_alignment*flock.Flock.num_bins_neighbors
    n_actions = flock.Flock.num_turns
    tol = 1e-8
    eps = 0.1

    n_episodes = 100
    episode_len = 1000

    flock_params = {
        'n_birds':200,
        'speed':0.5,
        'time_step':1,
        'perceptual_radius':2,
        'bounding_box':(-20,20),
        'birth_box':(-5,5),
        'random_seed':42
    }

    bbox_area = (flock_params['bounding_box'][1] - flock_params['bounding_box'][0])**2
    bird_density = flock_params['n_birds']/bbox_area
    print(f'Bird density = {bird_density}')

    qlearner = ql.QLearner(alpha, gamma, n_states, n_actions, flock_params['n_birds'], eps, tol)
    cost_episodes = qlearner.run_episodes(n_episodes, episode_len, flock_params)