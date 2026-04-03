#!/bin/bash

# Run the full Bernoulli sample-mean workflow:
# 1. generate unbiased samples
# 2. generate biased samples
# 3. glue the histograms into one distribution
# 4. view the results as PDFs in the plots folder

# Usage: bash run_all.sh

PROBABILITY=0.2 #success probability of the Bernoulli trials, you can change this to any value between 0 and 1

SAMPLE_SIZE=100 #number of samples inside the sample mean y=\overline{x}=1/n sum_{i=1}^n x_i, where n=sample_size. 
#Do not change this! If you want to change the sample size, you need to adapt the bias paramters, theta1, theta2 and theta3 as well.

echo "Running unbiased sampling with probability=${PROBABILITY} and sample_size=${SAMPLE_SIZE}"
cd sampling
bash run_bernoulli.sh "$PROBABILITY" "$SAMPLE_SIZE"

echo "Running biased sampling with probability=${PROBABILITY} and sample_size=${SAMPLE_SIZE}"
bash run_biased.sh "$PROBABILITY" "$SAMPLE_SIZE"

cd ..

mkdir -p ./plots

echo "Running glue distribution with probability=${PROBABILITY} and sample_size=${SAMPLE_SIZE}"
python3 glue_distribution.py --p "$PROBABILITY" --sample_size "$SAMPLE_SIZE"

echo "Workflow finished successfully."
