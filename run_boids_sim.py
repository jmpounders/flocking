import pickle

import flock
import boids

if __name__=='__main__':

    episode_len = 100

    flock_params = {
        'n_birds':500,
        'speed':0.5,
        'time_step':1,
        'bounding_box':(-20,20),
        'birth_box':(-5,5),
        'random_seed':42
    }

    bbox_area = (flock_params['bounding_box'][1] - flock_params['bounding_box'][0])**2
    bird_density = flock_params['n_birds']/bbox_area
    print(f'Bird density = {bird_density}')

    flck = flock.Flock(**flock_params)
    sim = boids.Boids(flck)
    output = sim.run_episode(episode_len=episode_len, save_trajectories=True)

    with open('sim_output.pickle', 'wb') as f:
        pickle.dump(output, f)