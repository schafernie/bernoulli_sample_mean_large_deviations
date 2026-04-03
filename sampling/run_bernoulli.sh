#!/bin/bash


mkdir -p ./../data

# Bash script to run the Bernoulli sample mean script with given parameters.
# Usage: bash run_bernoulli.sh [PROBABILITY] [SAMPLE_SIZE]

NUM_SAMPLES=100000 #number of samples

PROBABILITY=${1:-0.2} #Bernoulli success probability
SAMPLE_SIZE=${2:-100} #number of Bernoulli trials in each sample
RNG_SEED=$((RANDOM + 10000 * RANDOM)) #seed for random number generator

# create the data folder if it does not exist


# run the Python script with the provided arguments
python3 bernoulli_sample_mean_unbiased_sampling.py --num_samples $NUM_SAMPLES --sample_size $SAMPLE_SIZE --probability $PROBABILITY --rng_seed $RNG_SEED
