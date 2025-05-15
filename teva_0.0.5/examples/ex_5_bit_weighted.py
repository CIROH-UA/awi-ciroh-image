import numpy as np
import teva
# from teva import dnfea
# from teva.utilities import plotting
import logging
import matplotlib.pyplot as plt

n_observations = 1250
n_features = 5
visualize = False
output_logging_level = logging.INFO

observation_columns = []
for i in range(n_features):
    column = []
    for j in range(i + 1):
        column.append(np.random.rand(n_observations))
    for k in range(n_features - i):
        column.append(np.ones(n_observations))
    column = np.sum(np.vstack(column), axis=0)
    observation_columns.append(column)

# this method should demonstrate which features are most important
# leftmost row has no randomness, rightmost row is entirely random
observation_table = np.vstack(observation_columns).T
# classes are the floored sums of the observation table
classifications = np.sum(observation_table, axis=1, dtype=int)

fig, ax = plt.subplots()
axes = [column.tolist() for column in observation_table.T]# + [classifications.tolist()]
ax.stackplot(np.arange(n_observations).tolist(), *axes)
ax.plot(classifications, color="black", linewidth=0.5)
# ax.stackplot(*axes)
plt.show()

teva_alg = teva.TEVA(ccea_max_order=n_features,
                     ccea_offspring_per_gen=n_features,
                     ccea_num_new_pop=n_features,
                     ccea_total_generations=30,
                     ccea_n_age_layers=5,
                     # ccea_max_novel_order=4,
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

target_class = unique_classes[0]
#
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

# teva_alg._ccea_stats()

# print(teva_alg.get_archived_dnfs(True))
# print()
# print(teva_alg.get_archived_dnfs(False))
all_dnfs = teva_alg.get_all_archived_dnfs()
feature_frequency = np.zeros(n_features)
features = []
for cls in all_dnfs.keys():
    print(f"{cls}: {all_dnfs[cls][0].mask.shape[0]}")
    for dnf in all_dnfs[cls]:
        for cc in dnf.items:
            feature_frequency += cc.mask
            features.append(cc.items)

# feature_zip = list(zip(features))
print(feature_frequency)
# for feature in feature_zip:
#     feat_freq = feature[0].feature_set()
#     for feat in feature[1:]:
#         feat.feature_set()

teva_alg.export("cc_5_bit_weighted.xlsx", "dnf_5_bit_weighted.xlsx")

teva_alg.plot_all(plot_ccs=True,
                  plot_dnfs=True,
                  plot_contours=False,
                  single_plot=False)

plt.show()
# teva_alg.plot_all()
