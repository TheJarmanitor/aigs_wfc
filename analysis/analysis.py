# %% import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# %%

df = pd.read_excel("app_responses.xlsx")


def rename_column(col: str, conditions: dict) -> str:
    for cond, value in conditions.items():
        if cond in col:
            col = value
    return col


name_conditions = {
    "Very Difficult": "difficulty_in_guidance",
    "Local Patterns": "local_patterns",
    "Global Pattern": "global_pattern",
    "part of Dragon Warrior": "similarity_to_game",
}

df = df.dropna(axis=1)
df = df.drop([3], axis=0)
new_columns = [rename_column(col, name_conditions) for col in df.columns]
sufixes = ["A", "B", "C"]
for i in range(len(sufixes)):
    for name in name_conditions.values():
        new_columns[new_columns.index(name)] = name + "_" + sufixes[i]
df.columns = new_columns
# %%

df_answers = df.iloc[:, 3:]

# %%
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
    df_answers["difficulty_in_guidance_A"],
    df_answers["difficulty_in_guidance_B"],
    alternative="greater",
)
print(f"Difficulty A vs B: {res.pvalue:.3f}")
res = stats.wilcoxon(
    df_answers["local_patterns_A"],
    df_answers["local_patterns_B"],
    alternative="less",
)
print(f"Local A vs B: {res.pvalue:.3f}")
res = stats.wilcoxon(
    df_answers["global_pattern_A"],
    df_answers["global_pattern_B"],
    alternative="greater",
)
print(f"Global A vs B: {res.pvalue:.3f}")
res = stats.wilcoxon(
    df_answers["similarity_to_game_A"],
    df_answers["similarity_to_game_B"],
    alternative="greater",
)
print(f"Similarity A vs B: {res.pvalue:.3f}")
# %%
res = stats.wilcoxon(
    df_answers["difficulty_in_guidance_A"],
    df_answers["difficulty_in_guidance_C"],
    alternative="greater",
)
print(f"Difficulty A vs C: {res.pvalue:.3f}")
res = stats.wilcoxon(
    df_answers["local_patterns_A"],
    df_answers["local_patterns_C"],
    alternative="greater",
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
    alternative="greater",
)
print(f"Similarity A vs C: {res.pvalue:.3f}")

# %%
res = stats.wilcoxon(
    df_answers["difficulty_in_guidance_B"],
    df_answers["difficulty_in_guidance_C"],
    alternative="less",
)
print(f"Difficulty B vs C: {res.pvalue:.3f}")
res = stats.wilcoxon(
    df_answers["local_patterns_B"],
    df_answers["local_patterns_C"],
    alternative="less",
)
print(f"Local B vs C: {res.pvalue:.3f}")
res = stats.wilcoxon(
    df_answers["global_pattern_B"],
    df_answers["global_pattern_C"],
    alternative="less",
)
print(f"Global B vs C: {res.pvalue:.3f}")
res = stats.wilcoxon(
    df_answers["similarity_to_game_B"],
    df_answers["similarity_to_game_C"],
    alternative="less",
)
print(f"Similarity B vs C: {res.pvalue:.3f}")
