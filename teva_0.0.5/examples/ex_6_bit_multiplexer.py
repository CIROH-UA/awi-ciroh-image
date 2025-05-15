import numpy as np
import teva
# from teva import dnfea
# from teva.utilities import plotting
import logging
import matplotlib.pyplot as plt


def de2bi(d, n):
    """
    Convert decimal to binary representation.

    Parameters:
    - d: Decimal number or array of decimal numbers.
    - n: Number of bits for the binary representation.

    Returns:
    - Binary representation as a NumPy array.
    """
    # Ensure d is a NumPy array
    d = np.asarray(d)

    # Calculate powers of 2 up to n
    power = 2 ** np.arange(n)

    # Initialize binary representation array with zeros
    b = np.zeros_like(d)

    # Iterate over each element in d
    for i in range(len(d)):
        # Perform the conversion for each element
        b[i] = np.floor((d[i] % (2 * power)) / power)

    return b

def multiplexer(k, n_obs):
    n_bits = k + 2 ** k
    inputs = np.random.random((n_obs, n_bits))
    # address_bits = bin(list(np.arange(0, 2 ** k - 1)))
    address_bits = list(map(bin, range(2 ** k)))


n_observations = 1250
n_features = 5
visualize = True
output_logging_level = logging.INFO

multiplexer(2, 500)

observation_table = np.array(np.random.rand(n_observations, n_features) < 0.5)
classifications = np.sum(observation_table, axis=1) > n_features / 2.0

teva_alg = teva.TEVA(ccea_num_features=n_features,
                     ccea_offspring_per_gen=n_features,
                     ccea_num_new_pop=n_features,
                     ccea_total_generations=30,
                     ccea_best_fit=False,
                     ccea_n_age_layers=5,
                     ccea_gen_per_growth=3,
                     ccea_layer_size=n_features,
                     ccea_p_crossover=0.5,
                     ccea_p_wildcard=0.75,
                     ccea_p_mutation=1 / n_features,
                     ccea_tournament_size=3,
                     ccea_selective_mutation=False,
                     ccea_feature_sensitivity=False,
                     ccea_feature_sensitivity_threshold=-np.inf,
                     ccea_selection_exponent=3,
                     ccea_fitness_threshold=np.log10(1 / n_observations),
                     ccea_archive_bin_size=20,
                     ccea_max_novel_order=4,

                     dnfea_total_generations=60,
                     dnfea_gen_per_growth=3,
                     dnfea_n_age_layers=5,
                     dnfea_offspring_per_gen=20,
                     dnfea_p_crossover=0.5,
                     dnfea_p_targeted_mutation=0.2,
                     dnfea_p_targeted_crossover=0.25,
                     dnfea_tournament_size=3,
                     dnfea_p_union=0.5,
                     dnfea_p_intersection=0.0,
                     dnfea_record_best_fit=False,
                     dnfea_selection_exponent=3,
                     dnfea_max_ccs=12)
                     # dnfea_max_ccs=4)

unique_classes = teva_alg.fit(observation_table=observation_table,
                              classifications=classifications)

# target_class = unique_classes[0]

# teva_alg.run(target_class,
#              logfile_logging_level=logging.INFO,
#              output_logging_level=output_logging_level,
#              visualize=visualize,
#              print_timers=True)

teva_alg.run_all_targets(logfile_logging_level=logging.INFO,
                         output_logging_level=output_logging_level,
                         visualize=visualize,
                         print_timers=True)
#
# teva_alg.plot(target_class=target_class,
#               plot_ccs=True,
#               plot_dnfs=True,
#               plot_contours=True)


teva_alg.plot_all(plot_ccs=True,
                  plot_dnfs=True,
                  plot_contours=True,
                  single_plot=False)

plt.show()
# teva_alg.plot_all()
