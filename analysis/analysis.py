# %% import libraries
import pandas as pd
import numpy as np
# %% set up matplotlib specific stuff
import matplotlib.pyplot as plt



# %%

df  = pd.read_excel("app_responses.xlsx")
def rename_column(col: str, conditions: dict)->str:
    for cond, value in conditions.items():
        if cond in col:
            col = value
    return col

name_conditions = {
    "Very Difficult": "difficulty",
    "Local Patterns": "local_patterns",
    "Global Pattern": "global_pattern",
    "part of Dragon Warrior": "similarity"
}

df = df.dropna(axis=1)
new_columns = [rename_column(col, name_conditions) for col in df.columns]
sufixes = ["A", "B", "C"]
for i in range(len(sufixes)):
    for name in name_conditions.values():
        new_columns[new_columns.index(name)] = name + "_" + sufixes[i]
df.columns = new_columns
# %%

df_answers = df.iloc[:,3:]

# %%
fig, axes = plt.subplots(2,2)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42

for i, name in enumerate(name_conditions.values()):
    ax = axes.reshape(-1)[i]
    df_answers.boxplot(column=[col for col in df_answers.columns if name in col], ax=ax)
    labels = ["A", "B", "C"]
    ax.set_xticklabels(labels, ha='right')
    ax.set_title(name)

fig.suptitle("Distribution of survey responses by specific question")
fig.tight_layout()

plt.show()
plt.savefig("boxplot.pdf")
