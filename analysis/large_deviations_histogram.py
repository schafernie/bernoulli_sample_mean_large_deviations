"""
Histogram utilities for large-deviation reweighting.

This module stores histogram data together with the bias parameters used to
generate it, and provides routines to glue several overlapping histograms into
one log-density estimate. Bootstrap resampling is also included to estimate the
uncertainty of the glued distribution.
"""

import copy
import matplotlib.pyplot as plt
import numpy as np


class Histogram:
    """Store histogram data and basic derived quantities."""

    def __init__(self, in_values, in_bin_edges, in_parameter):
        self.parameter = copy.deepcopy(in_parameter)
        self.values = copy.deepcopy(in_values)
        self.bin_edges = copy.deepcopy(in_bin_edges)
        self.nr_bins = len(self.bin_edges) - 1
        self.counts = np.histogram(self.values, bins=self.bin_edges)[0]
        self.all_counts = len(self.values)

    def get_bin_center(self):
        """Return the centers of the histogram bins."""
        return (self.bin_edges[1 : self.nr_bins + 1] + self.bin_edges[: self.nr_bins]) / 2

    def bootstrap_counts(self, rng):
        """Draw a bootstrap sample and return its histogram counts."""
        bootstrap_sample = rng.choice(self.values, self.all_counts, replace=True)
        return np.histogram(bootstrap_sample, bins=self.bin_edges)[0]

    def boot_counts(self, rng):
        """Backward-compatible alias for :meth:`bootstrap_counts`."""
        return self.bootstrap_counts(rng)

    def print(self):
        """Print the associated bias parameters."""
        self.parameter.print()


class Parameter:
    """Store the bias parameters used for reweighting a histogram."""

    def __init__(self, theta1, theta2, theta3):
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.param2 = -1
        if self.theta2 > 0:
            self.param2 = self.theta2**2 / 2.0

    def set_parameter(self, theta1, theta2, theta3):
        """Update the parameter values in place."""
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.param2 = -1
        if self.theta2 > 0:
            self.param2 = self.theta2**2 / 2.0

    def print(self):
        """Print the parameter values."""
        print("ld parameter:")
        print(f"theta1: {self.theta1}\ntheta2: {self.theta2}\ntheta3: {self.theta3}")


def _find_reference_histogram_index(histograms):
    """Return the index of the unbiased histogram with theta1 close to zero."""
    for index, hist in enumerate(histograms):
        if abs(hist.parameter.theta1) < 1e-7:
            return index
    raise ValueError("No reference histogram with theta1 close to zero was found.")

def _reweighted_log_counts(log_counts, ys, parameter):
    """Apply the bias correction associated with one histogram."""
    reweighted = log_counts - ys * parameter.theta1
    if parameter.param2 > 0:
        reweighted += parameter.param2 * (ys - parameter.theta3) ** 2
    reweighted[np.isinf(reweighted)] = 0
    return reweighted


def _compute_pair_shift(counts_a, counts_b, values_a, values_b, shift_a):
    """Compute the relative shift between two overlapping histograms."""
    overlap_counts = np.min(np.stack((counts_a, counts_b), axis=0), axis=0)
    
    
    overlap_sum = np.sum(overlap_counts)
    if overlap_sum == 0:
        return shift_a
    return np.sum(overlap_counts * (values_a - values_b)) / overlap_sum + shift_a


def _order_histogram_indices(histograms):
    """Order histogram indices from negative theta1 through zero to positive theta1."""
   # print("hallo welt")
    reference_index = _find_reference_histogram_index(histograms)

    negative_indices = sorted(
        [index for index, hist in enumerate(histograms) if hist.parameter.theta1 < -1e-7],
        key=lambda index: (histograms[index].parameter.theta1, histograms[index].parameter.theta3),
    )
    positive_indices = sorted(
        [index for index, hist in enumerate(histograms) if hist.parameter.theta1 > 1e-7],
        key=lambda index: (histograms[index].parameter.theta1, histograms[index].parameter.theta3),
    )

    return negative_indices + [reference_index] + positive_indices


def order_histograms(histograms):
    """Sort histograms in place by theta1, then theta3."""
    ordered = [histograms[i] for i in _order_histogram_indices(histograms)]
    histograms[:] = ordered


def _compute_shifts(histograms):
    """Compute additive shifts that align all histograms on a common scale.

    Assumes histograms are already sorted (e.g. via order_histograms).
    """
    ys = histograms[0].get_bin_center()
    with np.errstate(divide="ignore"):
        log_counts = [np.log(hist.counts) for hist in histograms]

    shifts = np.zeros(len(histograms))
    for index in range(1, len(histograms)):
        values_a = _reweighted_log_counts(log_counts[index - 1], ys, histograms[index - 1].parameter)
        values_b = _reweighted_log_counts(log_counts[index], ys, histograms[index].parameter)
        shifts[index] = _compute_pair_shift(
            histograms[index - 1].counts,
            histograms[index].counts,
            values_a,
            values_b,
            shifts[index - 1],
        )

    return shifts


def _plot_reweighted_histograms(histograms):
    """Plot the reweighted and shifted histograms used during glueing."""
    ys = histograms[0].get_bin_center()
    shifts = _compute_shifts(histograms)

    _, ax_reweighted = plt.subplots()
    ax_reweighted.set_ylabel("reweighted histogram")
    ax_reweighted.set_xlabel(r"$y$")

    _, ax_shifted = plt.subplots()
    ax_shifted.set_ylabel("reweighted+shifted histograms")
    ax_shifted.set_xlabel(r"$y$")

    for index, hist in enumerate(histograms):
        with np.errstate(divide="ignore"):
            log_counts = np.log(hist.counts)
        reweighted= _reweighted_log_counts(log_counts, ys, hist.parameter)
        mask = hist.counts > 0
        ax_reweighted.plot(ys[mask], reweighted[mask], ls=" ", marker="x")
        ax_shifted.plot(ys[mask], reweighted[mask] + shifts[index], ls=" ", marker="x")


def ln_density(histograms, do_plot=True):
    """Glue histograms into one normalized log-density."""
    order_histograms(histograms)
    y = histograms[0].get_bin_center()
    with np.errstate(divide="ignore"):
        log_counts = [np.log(hist.counts) for hist in histograms]
    shifts = _compute_shifts(histograms)

    if do_plot:
        _plot_reweighted_histograms(histograms)

    log_density = np.full(len(y), -np.inf)

    for bin_index, y_value in enumerate(y):
        weighted_sum = 0.0
        total_count = 0

        for hist_index, hist in enumerate(histograms):
            count = hist.counts[bin_index]
            if count <= 0:
                continue

            value = (
                log_counts[hist_index][bin_index]
                - y_value * hist.parameter.theta1
                + shifts[hist_index]
            )
            if hist.parameter.param2 > 0:
                value += hist.parameter.param2 * (y_value - hist.parameter.theta3) ** 2
            if np.isinf(value):
                value = 0

            weighted_sum += count * value
            total_count += count
        if total_count > 0:
            log_density[bin_index] = weighted_sum / total_count

    log_density = log_density - np.max(log_density)
    #print(log_density)
    norm = np.trapz(np.exp(log_density), y)
    log_density = log_density - np.log(norm)
    print(f"normalization: {np.trapz(np.exp(log_density), y):.6f}")

    return y, log_density
