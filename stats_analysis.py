import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
import scipy.stats as stats

# Local config
DATA_DIR = "./diversity_analysis_results"
ALPHA = 0.05  # Significance level (of p-value) for hypothesis testing
METRIC = "coverages_unweighted"  # Or "coverages_weighted"


# Best p-value calculation for this experiment: no need of a normal distribution assumption
# and all runs are independent, so Mann-Whitney U test is a good choice for comparing two samples.
def mannwhitney_test(a, b):
    if np.array_equal(a, b):
        return math.inf, math.inf
    return stats.mannwhitneyu(a, b, alternative="two-sided")


# Correction for multiple testing: control the family-wise error rate (FWER)
# TODO: try appling Holm correction, which is less conservative than Bonferroni
def bonferroni_correction(p_values):
    m = len(p_values)
    return [min(p * m, 1.0) for p in p_values]


def main():

    stats_file = Path(DATA_DIR) / "coverages.json"
    if not stats_file.exists():
        print(f"Stats file not found: {stats_file}")
        return

    with open(stats_file, "r") as f:
        data = json.load(f)

    archives = list(data.keys())
    results = []

    # Confronto tutte le coppie
    for arch_a, arch_b in combinations(archives, 2):

        a = np.array(data[arch_a][METRIC])
        b = np.array(data[arch_b][METRIC])
        if a.size == 0 or b.size == 0:
            print(f"Warning: Empty data for {arch_a} or {arch_b} on {METRIC}, skipping.")
            continue

        u_stat, p_val = mannwhitney_test(a, b)

        results.append({"archive_A": arch_a, "archive_B": arch_b, "U_statistic": u_stat, "p_value": p_val})

    # Correzione multiple testing (Bonferroni)
    p_vals = [r["p_value"] for r in results]
    p_vals_corrected = bonferroni_correction(p_vals)

    for r, p_corr in zip(results, p_vals_corrected):
        r["p_value_corrected"] = p_corr

    # Salva risultati
    df_results = pd.DataFrame(results)
    output_file = Path(DATA_DIR) / f"mannwhitney_results_{METRIC}.csv"
    df_results.to_csv(output_file, index=False)

    print("\n=== Results ===")
    print(df_results)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
