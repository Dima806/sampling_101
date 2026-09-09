"""Cluster sampling and the grouped-data trap.

Rows are rarely independent. When they nest inside entities, two things follow. Sampling
whole entities (cluster sampling) is cheap but *less* precise than a simple random sample
of the same number of rows, because rows within an entity repeat information. And splitting
rows at random leaks entities across the split, which is the leakage bug that quietly
inflates every cross-validation score computed on grouped data.
"""

import numpy as np

from src.survey.simple import Estimate, simple_random_indices


def cluster_mean(
    population: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    rng: np.random.Generator,
) -> Estimate:
    """Estimate the population mean by sampling whole groups.

    Assumes equal group sizes, which the synthetic population guarantees. The standard
    error comes from the spread of the *group* means, not of the rows: with ``g`` groups
    sampled, the effective sample size is ``g``, however many rows each one contributes.
    """
    labels = np.unique(groups)
    total_groups = labels.size
    if not 2 <= n_groups <= total_groups:
        raise ValueError(f"need 2 <= n_groups <= {total_groups}, got {n_groups}")

    chosen = labels[simple_random_indices(total_groups, n_groups, rng)]
    group_means = np.array([population[groups == g].mean() for g in chosen])

    value = float(group_means.mean())
    fpc = max(1.0 - n_groups / total_groups, 0.0)
    se = float(np.sqrt(fpc * group_means.var(ddof=1) / n_groups))
    return Estimate(value=value, se=se, n=n_groups)


def naive_row_split(
    n_rows: int, test_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows at random, ignoring group structure. The convenient, wrong default."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    n_test = int(round(n_rows * test_fraction))
    test_idx = simple_random_indices(n_rows, n_test, rng)
    test_mask = np.zeros(n_rows, dtype=bool)
    test_mask[test_idx] = True
    return ~test_mask, test_mask


def group_train_test_split(
    groups: np.ndarray, test_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Split by entity, so every row of a group lands on the same side."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    labels = np.unique(groups)
    n_test_groups = max(int(round(labels.size * test_fraction)), 1)
    test_groups = labels[simple_random_indices(labels.size, n_test_groups, rng)]
    test_mask = np.isin(groups, test_groups)
    return ~test_mask, test_mask


def group_leakage_rate(groups: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray) -> float:
    """Fraction of test groups that also appear in the training set.

    Zero for an entity-aware split. Close to one for a random row split, which is the
    quantified form of "your test set already knows the answer".
    """
    test_groups = np.unique(groups[test_mask])
    if test_groups.size == 0:
        return 0.0
    train_groups = np.unique(groups[train_mask])
    return float(np.isin(test_groups, train_groups).mean())
