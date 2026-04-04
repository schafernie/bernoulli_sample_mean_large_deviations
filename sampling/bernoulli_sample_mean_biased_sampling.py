"""
Biased sampling of Bernoulli sample means using a Metropolis-Hastings-like algorithm.

Generates biased sample means of Bernoulli(p) variables using an exponential bia
controlled by parameters theta1, theta2, and theta3. Results are saved to disk for
use in gluing distributions
"""

import argparse
import numpy as np
from pathlib import Path

# Configuration parameters
OUTPUT_PATH = Path("./../data/")

def generate_biased_sample_means(num_samples: int, sample_size: int, probability: float, rng_seed: int,
                                 theta1: float, theta2: float, theta3: float, equilibration: int, delta: int) -> np.ndarray:
    """
    Generate biased sample means of sample size Bernoulli variables using Metropolis-Hastings like sampling.

    args:
        num_samples: number of sample means to generate.
        sample_size: size of each sample.
        probability: probability of success for bernoulli distribution.
        rng_seed: seed for random number generator.
        theta1: bias parameter 1.
        theta2: bias parameter 2.
        theta3: bias parameter 3.
        equilibration: equilibration steps.
        delta: thinning interval to decorrelate samples.

    returns:
        array of biased sample means
    """
    rng = np.random.default_rng(rng_seed)

    # parameter setup
    param2 = -1
    if theta2 > 0:
        param2 = theta2**2 / 2.
    else:
        theta2 = -1
        theta3 = -1

    total_steps = equilibration + delta * num_samples
    sample_means = np.zeros(total_steps)
    indexes = np.arange(sample_size, dtype=int)

    # initialization
    observations = rng.binomial(1, probability, size=sample_size)
    y = np.mean(observations)
    dy2 = (y - theta3)**2
    y_prop=0
    dy2_prop = 0
    sample_means[0] = y
    for n in range(1, total_steps):
        rand_indexes = rng.choice(indexes, sample_size, replace=True)
        observations_prop = rng.binomial(1, probability, size=sample_size)
        uniform01s = rng.uniform(0., 1., sample_size)
        for c in range(sample_size):
            index = rand_indexes[c]
            obs = observations[index]
            obs_prop = observations_prop[c]
            y_prop = y - (obs / sample_size) + (obs_prop / sample_size)
            delta1 = y_prop - y
            if param2 > 0:
                dy2_prop = (y_prop - theta3)**2
                delta2 = dy2_prop - dy2
                exponent = theta1 * delta1 - param2 * delta2
            else:
                exponent = theta1 * delta1
            if exponent > 0 or uniform01s[c] < np.exp(exponent):
                observations[index] = obs_prop
                y = y_prop                
                dy2 = dy2_prop
        sample_means[n] = y

    # return thinned samples after equilibration
    return sample_means[equilibration::delta]

def main():
    parser = argparse.ArgumentParser(description="generate and save biased bernoulli sample means.")
    parser.add_argument("--num_samples", type=int, default=10000, help="number of sample means to generate.")
    parser.add_argument("--sample_size", type=int, default=100, help="size of each sample.")
    parser.add_argument("--probability", type=float, default=0.2, help="probability of success for bernoulli distribution.")
    parser.add_argument("--rng_seed", type=int, default=50, help="seed for random number generator.")
    parser.add_argument("--theta1", type=float, default=-50, help="bias parameter 1.")
    parser.add_argument("--theta2", type=float, default=-1, help="bias parameter 2.")
    parser.add_argument("--theta3", type=float, default=-1, help="bias parameter 3.")
    parser.add_argument("--equilibration", type=int, default=100000, help="equilibration steps.")
    parser.add_argument("--delta", type=int, default=1, help="thinning interval.")

    args = parser.parse_args()

    # generate biased sample means
    sample_means = generate_biased_sample_means(args.num_samples, args.sample_size, args.probability, args.rng_seed,
                                                 args.theta1, args.theta2, args.theta3, args.equilibration, args.delta)

    # save sample means to file
    file_name = f"bernoulli_mean_biased_p{args.probability:.4f}_size{args.sample_size}_seed{args.rng_seed}_"
    file_name += f"1theta{args.theta1:.4f}_2theta{args.theta2:.4f}_3theta{args.theta3:.4f}_.dat"
    header = f"theta1: {args.theta1} , theta2: {args.theta2} , theta3: {args.theta3} \n"
    header += f"y samples (biased bernoulli means), (equilibration time: {args.equilibration} , delta: {args.delta} )"
    np.savetxt(OUTPUT_PATH / file_name, sample_means, header=header)

if __name__ == "__main__":
    main()        
        