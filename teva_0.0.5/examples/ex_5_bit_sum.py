import numpy as np
import teva
import logging
import matplotlib.pyplot as plt

n_observations = 800
n_features = 5
visualize = True
output_logging_level = logging.INFO

observation_table = (np.random.rand(n_observations, n_features) * 2).astype(int)
# classifications = np.sum(observation_table, axis=1) > n_features / 2.0
classifications = np.sum(observation_table, axis=1).astype(int)


teva_alg = teva.TEVA(ccea_max_order=n_features,
                     ccea_offspring_per_gen=n_features,
                     ccea_num_new_pop=n_features,
                     ccea_total_generations=30,
                     ccea_n_age_layers=5,
                     ccea_gen_per_growth=3,
                     ccea_layer_size=n_features,
                     ccea_archive_offspring_per_gen=25,
                     ccea_p_crossover=0.5,
                     ccea_p_wildcard=0.75,
                     ccea_p_mutation=1 / n_features,
                     ccea_tournament_size=3,
                     ccea_selective_mutation=False,
                     ccea_use_sensitivity=False,
                     ccea_sensitivity_threshold=1.25,
                     ccea_selection_exponent=3,
                     ccea_fitness_threshold=1 / n_observations,
                     ccea_archive_bin_size=20,

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
                     dnfea_selection_exponent=3,
                     dnfea_max_order=12,
                     dnfea_layer_size=20)
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
                         visualize=visualize)
#
# teva_alg.plot(target_class=target_class,
#               plot_ccs=True,
#               plot_dnfs=True,
#               plot_contours=True)

teva_alg.export("cc_5_bit_randomizer.xlsx", "dnf_5_bit_randomizer.xlsx")

teva_alg.plot_all(plot_ccs=True,
                  plot_dnfs=True,
                  plot_contours=True,
                  single_plot=False)

plt.show()
# teva_alg.plot_all()
