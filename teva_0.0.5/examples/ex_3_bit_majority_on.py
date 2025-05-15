import numpy as np
import teva
# from teva import dnfea
# from teva.utilities import plotting
import logging
import matplotlib.pyplot as plt
from teva.utilities import flatten_dict

n_observations = 1250
n_features = 3
visualize = False
output_logging_level = logging.INFO

observation_table = np.array(np.random.rand(n_observations, n_features) < 0.5)
classifications = np.sum(observation_table, axis=1) > n_features / 2.0

teva_alg = teva.TEVA(ccea_max_order=n_features,
                     ccea_offspring_per_gen=5,
                     ccea_num_new_pop=5,
                     ccea_total_generations=30,
                     ccea_n_age_layers=5,
                     # ccea_max_novel_order=4,
                     ccea_gen_per_growth=5,
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
                     # ccea_fitness_threshold=np.log10(1 / n_observations),
                     # ccea_fitness_threshold=-90,
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

teva_alg.export("cc_5_bit_majority.xlsx", "dnf_5_bit_majority.xlsx")

all_ccs = flatten_dict(teva_alg.get_all_archived_ccs())

all_dnfs = teva_alg.get_all_archived_dnfs()
feature_frequency = np.zeros(n_features)
features = []
for cls in all_dnfs.keys():
    print(f"{cls}: {all_dnfs[cls][0].mask.shape[0]}")
    for dnf in all_dnfs[cls]:
        for cc in dnf.items:
            feature_frequency += cc.mask
            features.append(cc.items)

false_dnfs = teva_alg.get_archived_dnfs(False)
cc_frequencies = []
for cls in false_dnfs.keys():
    for dnf in false_dnfs[cls]:
        cc_frequencies.append(dnf.mask)

teva_alg.plot_all(plot_ccs=True,
                  plot_dnfs=True,
                  plot_contours=True,
                  single_plot=False)

plt.show()
# teva_alg.plot_all()
