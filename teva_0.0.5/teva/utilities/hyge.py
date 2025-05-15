""" The Hypergeometric Mass Function """
from typing import Callable
import numpy as np

def hygepmf(k: int, N: int, K: int, n: int) -> float:
    """
    Calculates the hypergeometric probability mass function, much like
    scipy.stats.hypergeom.pmf() or scipy.stats.hypergeom.logpmf(),
    hyper except it returns base 10 logarithm of the probability mass function
    evaluated at k.

    Based on the Matlab function ``hygepmf()`` (John Hanley, 2018)

    .. note::

        The hypergeometric distribution is a discrete probability distribution
        that describes the probability of :math:`k` successes in :math:`n`
        draws, without replacement, from a finite population of size :math:`N`
        that contains exactly :math:`K` objects with that feature, wherein each
        draw is either a success or a failure.

        The probability mass function is given by

        .. math::

            p_X(k) = \\Pr(X = k) = \\frac{\\binom{K}{k}\\binom{N-K}{n-k}}{\\binom{N}{n}}.

        Using the quotient rule, the logarithm of the probability mass function
        is given by

        .. math::

                \\log_10(p_X(k)) = ( \\log_10(\\binom{K}{k}) + \\log_10(\\binom{N-K}{n-k})) - \\log_10(\\binom{N}{n}).

    .. seealso::

        ``scipy.stats.hypergeom.pmf``

        Module :py:mod:`scipy`

        `Hypergeometric Distribution` <https://en.wikipedia.org/wiki/Hypergeometric_distribution>

    :param k: number of observed successes
    :param n: number of draws
    :param K: number of success states in the population
    :param N: population size

    :return: the base 10 logarithm of the probability mass function evaluated at k

    :raises: ArithmeticError
    """
    # ensure input arguments are integers
    if not isinstance(k, int) and not np.issubdtype(type(k), np.integer):
        raise AttributeError("k must be an integer")
    if not isinstance(n, int) and not np.issubdtype(type(k), np.integer):
        raise AttributeError("n must be an integer")
    if not isinstance(K, int) and not np.issubdtype(type(k), np.integer):
        raise AttributeError("K must be an integer")
    if not isinstance(N, int) and not np.issubdtype(type(k), np.integer):
        raise AttributeError("N must be an integer")

    # enforce function bounds
    if N < 0:
        raise ArithmeticError(f"N must be an integer greater than or equal to 0: N={N}")

    if K > N:
        raise ArithmeticError(f"K must be an integer less than or equal to N: N={N}, K={K}")

    if n > N:
        raise ArithmeticError(f"n must be an integer less than or equal to N: N={N}, n={n}")

    # Separate equation into three parts for readability
    numerator_combination_1 = np.divide(np.arange(K - k + 1, K + 1), np.arange(1, k + 1))
    numerator_combination_2 = np.divide(np.arange(N - K - (n - k) + 1, N - K + 1), np.arange(1, n - k + 1))
    denominator_combination = np.divide(np.arange(N - n + 1, N + 1), np.arange(1, n + 1))

    # Take the log10 of each part
    try:
        log_numerator_combination_1 = np.log10(numerator_combination_1)
        log_numerator_combination_2 = np.log10(numerator_combination_2)
        log_denominator_combination = np.log10(denominator_combination)
    except ZeroDivisionError:
        print("Zero division")
        return np.nan

    # Find the numerator and denominator of the final logarithm
    numerator = np.sum(log_numerator_combination_1) + np.sum(log_numerator_combination_2)
    denominator = np.sum(log_denominator_combination)

    # get the log base 10 of the probability
    log_probability = numerator - denominator if denominator != 0 else numerator

    return log_probability

def hygcontours(classes: np.ndarray,
                target_class,
                fit_interval: int,
                fitness_function: Callable = hygepmf) -> (np.ndarray, np.ndarray):
    """ Helps create contour lines for the hypergeometric fitness function.

    :param classes: All unique classes from the input data
    :param target_class: The targeted class
    :param fit_interval: The interval of contour fitness lines
    :param fitness_function: The fitness function to use (default: :function:`hygepmf()`)

    :return: An array of fitness levels, an array of contour boundaries
    """
    n_obs = classes.shape[0]
    n_target = int(np.sum(classes == target_class))

    min_fit = fitness_function(n_target, n_obs, n_target, n_target)

    # cover_values = np.arange(1, n_target)
    all_ppv = np.full(shape=(n_target, n_obs), fill_value=np.nan)
    all_cov = np.full(shape=(n_target, n_obs), fill_value=np.nan)
    all_fitness = np.full(shape=(n_target, n_obs), fill_value=np.nan)

    for i in range(n_target):
        min_j = i + 1
        max_j = round(min_j * n_obs / n_target)

        for j in range(i, max_j):
            all_ppv[i, j] = min_j / (j + 1)
            all_cov[i, j] = min_j / n_target
            all_fitness[i, j] = fitness_function(min_j, n_obs, n_target, j + 1)

    # Create contour lines
    # determine the minimum contour level
    min_level = np.ceil(min_fit/fit_interval) * fit_interval

    # create a fitness bias
    fitness_levels = np.linspace(min_level, -fit_interval, np.arange(min_level, fit_interval, fit_interval).shape[0])
    fitness_levels = fitness_levels.astype(np.integer)
    # boundaries = {}
    # for level in fitness_levels:
    #     boundaries[level] = {}

    # determine minimum fitness for each row
    min_row_fitness = np.nanmin(all_fitness, axis=1)

    # Calculate the 100% PPV for every possible class coverage
    ppv_100_fit = []
    for i in range(n_target):
        ppv_100_fit.append(fitness_function(i, n_obs, n_target, i))

    boundaries: list[dict] = []
    for i in range(fitness_levels.shape[0]):
        start_row = np.nonzero(fitness_levels[i] > min_row_fitness)[0][0]

        lb_list = np.nonzero(ppv_100_fit > fitness_levels[i])[0]
        ub_list = np.nonzero(ppv_100_fit < fitness_levels[i])[0]

        if len(lb_list) == 0 or len(ub_list) == 0:
            continue

        lb_col = lb_list[-1]
        ub_col = ub_list[0]

        bound_cov = [(ub_col / n_target + lb_col / n_target) / 2.0]
        ub_ppv = [1]
        lb_ppv = [1]
        mean_ppv = [1]

        for j in range(start_row, all_ppv.shape[0]):
            lb_col = np.nonzero(all_fitness[j] > fitness_levels[i])[0][0]
            ub_col = np.nonzero(all_fitness[j] < fitness_levels[i])[0][-1]

            bound_cov.append(all_cov[j][ub_col])
            ub_ppv.append(all_ppv[j][ub_col])
            lb_ppv.append(all_ppv[j][lb_col])
            mean_ppv.append((ub_ppv[-1] + lb_ppv[-1]) / 2.0)

        boundaries.append({
            "bound_coverage": bound_cov,
            "upper_bound_ppv": ub_ppv,
            "lower_bound_ppv": lb_ppv,
            "mean_bound_ppv": mean_ppv
        })

    return fitness_levels, boundaries
