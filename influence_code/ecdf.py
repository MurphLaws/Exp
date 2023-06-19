from scipy.stats import spearmanr
from statsmodels.distributions.empirical_distribution import ECDF


def ecdf(influence_matrix_1, influence_matrix_2):
    assert influence_matrix_1.shape == influence_matrix_2.shape

    # The function assumes that rows represent the train sample ids and
    # columns the test sample ids. The correlations are computed column-wise.
    # This is because this structure implies the many-to-one relation, i.e., many train examples (or groups)
    # to one test example

    corr_vals = {}
    p_value_threshold = 0.05

    for col in influence_matrix_1.columns:
        spearman_r, pval = spearmanr(
            influence_matrix_1.loc[:, col], influence_matrix_2.loc[:, col]
        )
        if pval <= p_value_threshold:
            corr_vals[col] = spearman_r
    print(
        "Statistically significant in rows count:",
        len(corr_vals) / influence_matrix_1.shape[1],
    )
    ecdf = ECDF(list(corr_vals.values()))


def plot_ecdf(ecdf_arr):
    # TODO
    # please see the plot in the slides and try to replicate it
    # using simple matplotlib line plot
    pass