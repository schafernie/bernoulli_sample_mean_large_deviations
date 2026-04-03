#!/bin/bash

# Bash script to run the biased Bernoulli sample mean script.
# Usage: bash run_biased.sh [PROBABILITY] [SAMPLE_SIZE]

# create the data folder if it doesn't exist
mkdir -p ./../data

# default parameters
NUM_SAMPLES=10000
SAMPLE_SIZE=${2:-100}
PROBABILITY=${1:-0.2}
RNG_SEED=$((RANDOM + 10000 * RANDOM))
THETA2=-1
THETA3=-1
EQUILIBRATION=1000
DELTA=1

# loop over theta1 values
for THETA1 in -500 -400 -300 -200 -100 100 200 300 400 500
do
    echo "running with theta1 = $THETA1 and rng_seed = $RNG_SEED"
    python3 bernoulli_sample_mean_biased_sampling.py \
        --num_samples $NUM_SAMPLES \
        --sample_size $SAMPLE_SIZE \
        --probability $PROBABILITY \
        --rng_seed $RNG_SEED \
        --theta1 $THETA1 \
        --theta2 $THETA2 \
        --theta3 $THETA3 \
        --equilibration $EQUILIBRATION \
        --delta $DELTA

    RNG_SEED=$((RNG_SEED + 1))
done