# Bernoulli Sample Mean: Large-Deviations Sampling

Estimates the full probability distribution of the sample mean of Bernoulli random variables using a combination of direct Monte Carlo sampling and importance sampling with exponential bias weights. The sample mean is defined as

$$y =\overline{x}= \frac{1}{n} \sum_{i=1}^n x_i$$

where $x_i$ with $i=1,...,n$ are independent Bernoulli distributed random variables with PMF

$$P(x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}.$$

This approach efficiently recovers the distribution in the tails where direct sampling would be prohibitively expensive. For details on the method see: L. Münster, A. K. Hartmann and M. Weigel, Phys. Rev. E 110, 054112 (2024), https://doi.org/10.1103/PhysRevE.110.054112 

## Overview

The workflow has three stages:

1. **Unbiased sampling** — direct Monte Carlo to estimate the bulk of the distribution
2. **Biased sampling** — MCMC with exponential tilt to probe rare-event regions across the full range of the sample mean
3. **Distribution glueing** — histogram reweighting to merge all overlapping distributions into a single log-density estimate, then comparison against the exact Bernoulli PMF

## Quickstart

```bash
bash run_all.sh
```

Default parameters: Bernoulli probability `p=0.2`, sample size `n=100`.

## File Structure

```
bernoulli_sample_mean/
├── run_all.sh                                         # Main Bash file
├── glue_distribution.py                               # Merges histograms, generates plots
├── sampling/
│   ├── run_bernoulli.sh                               # Runs unbiased sampling
│   ├── run_biased.sh                                  # Runs biased sampling (10 theta1 values)
│   ├── bernoulli_sample_mean_unbiased_sampling.py     # Direct Monte Carlo sampler
│   └── bernoulli_sample_mean_biased_sampling.py       # MCMC importance sampler
├── analysis/
│   └── large_deviations_histogram.py                  # Histogram and glueing utilities
├── data/                                              # Generated sample data (.dat files)
└── plots/                                             # Output plots (.pdf files)
```

## Running Individual Steps

```bash
# Unbiased sampling
cd sampling && bash run_bernoulli.sh 0.2 100

# Biased sampling (10 theta1 values: -500 to 500 in steps of 100)
bash run_biased.sh 0.2 100

# Glueing (after samples exist)
cd .. && python3 glue_distribution.py --p 0.2 --sample_size 100
```

## Parameters

**`bernoulli_sample_mean_unbiased_sampling.py`**

| Argument | Default | Description |
|---|---|---|
| `--num_samples` | 100000 | Number of sample means to generate |
| `--sample_size` | 100 | Bernoulli trials per sample |
| `--probability` | 0.2 | Bernoulli success probability |
| `--rng_seed` | auto | Random seed |

**`bernoulli_sample_mean_biased_sampling.py`**

| Argument | Default | Description |
|---|---|---|
| `--num_samples` | 10000 | Samples to generate |
| `--sample_size` | 100 | Trials per sample |
| `--probability` | 0.2 | Success probability |
| `--theta1` | -50 | Primary exponential bias parameter |
| `--theta2` | -1 | Secondary bias parameter |
| `--theta3` | -1 | Tertiary bias parameter |
| `--equilibration` | 100000 | MCMC equilibration steps |
| `--delta` | 10 | Thinning interval (decorrelation) |
| `--rng_seed` | auto | Random seed |

## Outputs

**Data files** (in `data/`):
- `bernoulli_mean_unbiased_p{p}_size{n}_seed{seed}_.dat` — unbiased sample means
- `bernoulli_mean_biased_p{p}_size{n}_seed{seed}_1theta{theta1}_2theta{theta2}_3theta{theta3}_.dat` — biased MCMC samples per theta1 value
- `sampled_distribution_p{p}_size{n}_.dat` — final glued log-density (two columns: sample mean, log density)

**Plots** (in `plots/`):
- `sample_trajectories_*.pdf` — time series of all sampling runs showing coverage across the distribution
- `histograms_*.pdf` — raw histogram counts per run (log scale), before reweighting
- `glued_distribution_*.pdf` — comparison of exact PMF, unbiased sampling, and glued biased sampling

## Method

The bias function has the form `exp(theta1 * y + theta2 / 2 * (y-theta3)**2 )`. Running with ten values of theta3 spanning -500 to 500 forces the MCMC to explore different regions of the sample-mean range, including the far tails. The glueing step computes pairwise shifts between overlapping histograms, applies bias corrections, and averages overlapping regions to produce a log-density estimate.

Exact PMF values are computed in log space using the log-gamma function for numerical stability.

## Dependencies

Requires Python 3, matplotlib, numpy and Bash.

## License

MIT License — see [LICENSE](LICENSE) for details.

Copyright (C) 2026 Lambert Münster
