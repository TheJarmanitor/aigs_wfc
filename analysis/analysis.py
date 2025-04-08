# %% import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import pymc as pm
import arviz as az


# %%

df = pd.read_excel("app_responses.xlsx")


def rename_column(col: str, conditions: dict) -> str:
    for cond, value in conditions.items():
        if cond in col:
            col = value
    return col


name_conditions = {
    "easy to guide": "easiness_in_guidance",
    "Local Patterns": "local_patterns",
    "Global Pattern": "global_pattern",
    "Dragon Warrior": "similarity_to_game",
}

df = df.dropna(axis=1)
new_columns = [rename_column(col, name_conditions) for col in df.columns]
sufixes = ["A", "B", "C"]
for i in range(len(sufixes)):
    for name in name_conditions.values():
        new_columns[new_columns.index(name)] = name + "_" + sufixes[i]
df.columns = new_columns

# %%


# %%

df_answers = df.iloc[:, 8:]
fig, axes = plt.subplots(2, 2)
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.fonttype"] = 42

for i, name in enumerate(name_conditions.values()):
    ax = axes.reshape(-1)[i]
    df_answers.boxplot(column=[col for col in df_answers.columns if name in col], ax=ax)
    labels = ["A", "B", "C"]
    ax.set_xticklabels(labels, ha="right")
    ax.set_title(name.replace("_", " "))

fig.suptitle("Distribution of survey responses by specific question")
fig.tight_layout()

plt.show()
fig.savefig("boxplot.pdf")
# %% Data is non parametric with same population, wilcoxon rank sign test is used.

res = stats.wilcoxon(
    df_answers["easiness_in_guidance_A"],
    df_answers["easiness_in_guidance_B"],
    correction=True,
    # alternative="less",
)
print(f"Difficulty A vs B: {res.pvalue:.3f}")
res = stats.wilcoxon(
    df_answers["local_patterns_A"],
    df_answers["local_patterns_B"],
    correction=True,
    alternative="less",
)
print(f"Local A vs B: {res.pvalue:.3f}")
res = stats.wilcoxon(
    df_answers["global_pattern_A"],
    df_answers["global_pattern_B"],
    correction=True,
    alternative="greater",
)
print(f"Global A vs B: {res.pvalue:.3f}")
res = stats.wilcoxon(
    df_answers["similarity_to_game_A"],
    df_answers["similarity_to_game_B"],
    correction=True,
    # alternative="less",
)
print(f"Similarity A vs B: {res.pvalue:.3f}")
# %%
res = stats.wilcoxon(
    df_answers["easiness_in_guidance_A"],
    df_answers["easiness_in_guidance_C"],
    # alternative="less",
)
print(f"Difficulty A vs C: {res.pvalue:.3f}")
res = stats.wilcoxon(
    df_answers["local_patterns_A"],
    df_answers["local_patterns_C"],
    alternative="Greater",
)
print(f"Local A vs C: {res.pvalue:.3f}")
res = stats.wilcoxon(
    df_answers["global_pattern_A"],
    df_answers["global_pattern_C"],
    alternative="less",
)
print(f"Global A vs C: {res.pvalue:.3f}")
res = stats.wilcoxon(
    df_answers["similarity_to_game_A"],
    df_answers["similarity_to_game_C"],
    # alternative="less",
)
print(f"Similarity A vs C: {res.pvalue:.3f}")

# %%
res = stats.wilcoxon(
    df_answers["easiness_in_guidance_B"],
    df_answers["easiness_in_guidance_C"],
    # alternative="less",
)
print(f"Difficulty B vs C: {res.pvalue:.3f}")
res = stats.wilcoxon(
    df_answers["local_patterns_B"],
    df_answers["local_patterns_C"],
    # alternative="less",
)
print(f"Local B vs C: {res.pvalue:.3f}")
res = stats.wilcoxon(
    df_answers["global_pattern_B"],
    df_answers["global_pattern_C"],
    # alternative="less",
)
print(f"Global B vs C: {res.pvalue:.3f}")
res = stats.wilcoxon(
    df_answers["similarity_to_game_B"],
    df_answers["similarity_to_game_C"],
    # alternative="less",
)
print(f"Similarity B vs C: {res.pvalue:.3f}")


# %%

with pm.Model() as local_model:
    # Cauchy prior for effect size
    d = pm.Cauchy('d', alpha=0, beta=1, size=3)
    data_A = pm.Data("A", df_answers["local_patterns_A"].values)
    data_B = pm.Data("B", df_answers["local_patterns_B"].values)
    data_C = pm.Data("C", df_answers["local_patterns_C"].values)

    diff_AB = pm.Deterministic("diff_AB", data_A - data_B)
    diff_AC = pm.Deterministic("diff_AC", data_A - data_C)
    diff_BC = pm.Deterministic("diff_BC", data_B - data_C)

    lh_AB = pm.Normal('lh_AB', mu=d[0], sigma=1, observed=diff_AB)
    lh_AC = pm.Normal('lh_AC', mu=d[1], sigma=1, observed=diff_AC)
    lh_BC = pm.Normal('lh_BC', mu=d[2], sigma=1, observed=diff_BC)

    # Sampling
    trace_local = pm.sample(2000, tune=1000, idata_kwargs={"log_likelihood": True})
# %%
#

az.summary(trace_local, var_names=["d"])
# %%
az.plot_posterior(trace_local, var_names=['d'])
plt.show()
plt.tight_layout()
# print(f"Probability Version A > B: {np.mean(trace.posterior['d'] > 0):.2%}")
# %%

az.plot_forest(trace_local, var_names=["d"])

# %%
with pm.Model() as global_model:
    # Cauchy prior for effect size
    d = pm.Cauchy('d', alpha=0, beta=1, size=3)
    data_A = pm.Data("A", df_answers["global_pattern_A"].values)
    data_B = pm.Data("B", df_answers["global_pattern_B"].values)
    data_C = pm.Data("C", df_answers["global_pattern_C"].values)

    diff_AB = pm.Deterministic("diff_AB", data_A - data_B)
    diff_AC = pm.Deterministic("diff_AC", data_A - data_C)
    diff_BC = pm.Deterministic("diff_BC", data_B - data_C)

    lh_AB = pm.Normal('lh_AB', mu=d[0], sigma=1, observed=diff_AB)
    lh_AC = pm.Normal('lh_AC', mu=d[1], sigma=1, observed=diff_AC)
    lh_BC = pm.Normal('lh_BC', mu=d[2], sigma=1, observed=diff_BC)


    trace_global = pm.sample(2000, tune=1000, idata_kwargs={"log_likelihood": True})
# %%

az.summary(trace_global, var_names=["d"])

# %%
az.plot_forest(trace_global, var_names=["d"])
