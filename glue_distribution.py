"""
Glue biased and unbiased Bernoulli sample-mean histograms into one distribution.
"""

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analysis import large_deviations_histogram as ld


plt.close("all")

DATA_PATH = Path("./data")
PLOTS_PATH = Path("./plots")

# Physical and simulation parameters used to identify the data files
P = 0.2
SAMPLE_SIZE = 100
EQUILIBRATION_TIME = 0

# Common binning used for all histograms before they are glued together
BIN_START = -0.005
BIN_END = 1.01
BIN_WIDTH = 0.01
BIN_EDGES = np.arange(BIN_START, BIN_END, BIN_WIDTH)

# Output options
SAVE_DATA = True
SHOW_PLOTS = True


def log_bernoulli_sample_mean_pmf(k, n, p):
    """Log PMF of observing k successes in n Bernoulli(p) trials."""
    log_binom = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return log_binom + k * math.log(p) + (n - k) * math.log(1 - p)


def read_bias_parameters(file_path):
    """Read theta1, theta2, theta3 from the first line of a biased data file."""
    first_line = file_path.read_text().splitlines()[0]

    theta1 = float(re.search(r"(?<=theta1: )[-+]?(?:\d*\.\d+|\d+)", first_line)[0])
    theta2 = float(re.search(r"(?<=theta2: )[-+]?(?:\d*\.\d+|\d+)", first_line)[0])
    theta3 = float(re.search(r"(?<=theta3: )[-+]?(?:\d*\.\d+|\d+)", first_line)[0])

    return theta1, theta2, theta3


def load_values(file_paths, skip=0):
    """Load and concatenate values from a list of files."""
    all_values = []

    for file_path in file_paths:
        values = np.loadtxt(file_path)
        # For biased runs we can discard an initial equilibration window if needed.
        values = values[skip:]
        if len(values) > 0:
            all_values.append(values)

    if not all_values:
        return np.array([])

    return np.concatenate(all_values)


def main():
    """Run the histogram glueing workflow."""

    parser = argparse.ArgumentParser(
        description="Glue biased and unbiased Bernoulli sample-mean histograms into one distribution."
    )
    parser.add_argument("--p", type=float, default=P, help="Bernoulli success probability.")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=SAMPLE_SIZE,
        help="Number of Bernoulli trials in each sample.",
    )
    args = parser.parse_args()

    p = args.p
    sample_size = args.sample_size

    PLOTS_PATH.mkdir(exist_ok=True)

    # First discover which biased parameter combinations are available on disk.
    biased_files = DATA_PATH.glob(f"bernoulli_mean_biased_p{p:.4f}_size{sample_size}_*_.dat")

    parameter_sets = set()
    for file_path in biased_files:
        parameter_sets.add(read_bias_parameters(file_path))

        
    print("biased parameter sets found:")
    for theta1, theta2, theta3 in parameter_sets:
        print(f"theta1={theta1}, theta2={theta2}, theta3={theta3}")


    # Start with the unbiased reference data.
    histograms = []

    unbiased_files = DATA_PATH.glob(f"bernoulli_mean_unbiased_p{p:.4f}_size{sample_size}_seed*_.dat")
 
    unbiased_values = load_values(unbiased_files)

    if len(unbiased_values) > 0:
        mean = np.mean(unbiased_values)
        err = np.std(unbiased_values) / np.sqrt(len(unbiased_values))
        print(f"average sample mean from unbiased data: {mean:.6f} +- {err:.6f}")

        unbiased_histogram = ld.Histogram(
            unbiased_values,
            BIN_EDGES,
            ld.Parameter(0, -1, -1),
        )
        histograms.append(unbiased_histogram)


    # Then add one histogram for each biased ensemble.
    for theta1, theta2, theta3 in parameter_sets:
        biased_files = sorted(
            DATA_PATH.glob(
                f"bernoulli_mean_biased_p{p:.4f}_size{sample_size}_seed*_"
                f"1theta{theta1:.4f}_2theta{theta2:.4f}_3theta{theta3:.4f}_.dat"
            )
        )

        biased_values = load_values(biased_files, skip=EQUILIBRATION_TIME)
        if len(biased_values) == 0:
            continue

        histogram = ld.Histogram(
            biased_values,
            BIN_EDGES,
            ld.Parameter(theta1, theta2, theta3),
        )
        histograms.append(histogram)

    if not histograms:
        raise FileNotFoundError("No matching data files found in data/.")

    sample_counts = [len(histogram.values) for histogram in histograms]
    print(f"loaded {len(histograms)} histograms")
    print(f"maximum number of samples: {max(sample_counts)}")
    print(f"minimum number of samples: {min(sample_counts)}")
    
    ld.order_histograms(histograms)

    # Plot the raw sampled values to see which region each run explores.
    fig, ax = plt.subplots()
    for histogram in histograms:
        theta1 = histogram.parameter.theta1
        theta2 = histogram.parameter.theta2
        theta3 = histogram.parameter.theta3
        label = f"theta1={theta1}, theta2={theta2}, theta3={theta3}"
        ax.plot(histogram.values, label=label)

    ax.set_xlabel("mc step")
    ax.set_ylabel(r"$y=\overline{x}$")
    ax.set_title("sample trajectories")
    ax.set_xlim(0,2000)
    ax.legend(fontsize="small")
    fig.savefig(
        PLOTS_PATH / f"sample_trajectories_p{p:.4f}_size{sample_size}_.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


    # Plot the unglued histogram counts before combining them.
    fig, ax = plt.subplots()
    bin_centers = histograms[0].get_bin_center()

    for histogram in histograms:
        theta1 = histogram.parameter.theta1
        theta2 = histogram.parameter.theta2
        theta3 = histogram.parameter.theta3
        label = f"theta1={theta1}, theta2={theta2}, theta3={theta3}"
        ax.plot(bin_centers, histogram.counts, marker="o", ls="-", label=label)

    ax.set_yscale("log")
    ax.set_xlabel(r"$y=\overline{x}$")
    ax.set_ylabel("histogram count")
    ax.set_title("biased and unbiased histograms")
    ax.legend(fontsize="small")
    fig.savefig(
        PLOTS_PATH / f"histograms_p{p:.4f}_size{sample_size}_.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


    # Combine the overlapping histograms into a single log-density estimate.
    sample_means, log_density = ld.ln_density(histograms, SHOW_PLOTS)

    if SAVE_DATA:
        output_path = DATA_PATH / f"sampled_distribution_p{p:.4f}_size{sample_size}_.dat"
        output_data = np.column_stack((sample_means, log_density))
        np.savetxt(output_path, output_data, header="mean , log density")
        print(f"saved glued distribution to {output_path}")


    # Compare the glued result with the exact Bernoulli sample-mean distribution.
    support = np.arange(sample_size + 1)
    exact_log_pmf = np.empty(len(support))

    for i, k in enumerate(support):
        exact_log_pmf[i] = log_bernoulli_sample_mean_pmf(k, sample_size, p)

    support = support / sample_size
    normalization = np.trapezoid(np.exp(exact_log_pmf), support)
    exact_log_pmf = exact_log_pmf - np.log(normalization)
    
    with np.errstate(divide="ignore"):
        standard_log_density = np.log(unbiased_histogram.counts/(unbiased_histogram.all_counts*BIN_WIDTH))

    fig, ax = plt.subplots()
    ax.plot(support, exact_log_pmf, ls=" ", marker="o",markerfacecolor="none", label="exact",color="black")
    ax.plot(bin_centers, standard_log_density, ls=" ", marker="s",markerfacecolor="none", label="unbiased sampling",color="blue")
    ax.plot(sample_means, log_density, ls=" ", marker="x", label="biased sampling",color="red")
    ax.set_xlabel(r"$y=\overline{x}$")
    ax.set_ylabel(r"$\ln P(y)$")
    ax.set_title("glued distribution vs exact result")
    ax.legend()
    fig.savefig(
        PLOTS_PATH / f"glued_distribution_p{p:.4f}_size{sample_size}_.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()

