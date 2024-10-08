
import pandas as pd
from tabulate import tabulate

dir = "knapsack/10/2/test/"
filename = "RO_result_5000.csv"


alg_name_order = ["Ellips.","kNN","DCC","IDCC","PTC-B","PTC-E"]
df = pd.read_csv(dir+filename)


# preprocess the alg_name column, i.e., replace some algorithms' name with shorter name
for i in range(len(df)):
    if df.loc[i,"alg_name"][:3] == "kNN":
        df.loc[i,"alg_name"] = "kNN"
    elif df.loc[i,"alg_name"][:8] == "LUQ-quan":
        df.loc[i,"alg_name"] = "PTC-B"
    elif df.loc[i,"alg_name"][:8] == "LUQ-norm":
        df.loc[i,"alg_name"] = "PTC-E"
    elif df.loc[i,"alg_name"][:4] == "IDCC":
        df.loc[i,"alg_name"] = "IDCC"
    elif df.loc[i,"alg_name"][:3] == "DCC":
        df.loc[i,"alg_name"] = "DCC"
    elif df.loc[i,"alg_name"][:9] == "ellipsoid":
        df.loc[i,"alg_name"] = "Ellips."

# preprocess the mean_coverage column, only save to 2 decimal places
df["mean_coverage"] = df["mean_coverage"].apply(lambda x: round(x,2))

# preprocess the mean_VaR column, only save to integer
df["mean_VaR"] = df["mean_VaR"].apply(lambda x: round(x))

# preprocess the std_VaR column, only save to integer
df["std_VaR"] = df["std_VaR"].apply(lambda x: round(x))

# preprocess the std_coverage column, only save to 2 decimal places
df["std_coverage"] = df["std_coverage"].apply(lambda x: round(x,2))

# add a minus sign to the mean_VaR column if dir is start with "knapsack"
if dir[:8] == "knapsack":
    df["mean_VaR"] = df["mean_VaR"].apply(lambda x: -x)

# convert to required format
table_left = pd.pivot_table(df, values='mean_VaR', index='alpha', columns='alg_name')
# resort the column of the dataframe by matching "alg_name" with alg_name_order
table_left = table_left[alg_name_order]

"""
if dir[:8] == "shortest":
    # mean_VaR of "DCC" or "IDCC" should minus 250
    table_left["DCC"] = table_left["DCC"] - 250
    table_left["IDCC"] = table_left["IDCC"] - 250
"""

table_right = pd.pivot_table(df, values='mean_coverage', index='alpha', columns='alg_name')
# resort the column of the dataframe by matching "alg_name" with alg_name_order
table_right = table_right[alg_name_order]

table_std_left = pd.pivot_table(df, values='std_VaR', index='alpha', columns='alg_name')
# resort the column of the dataframe by matching "alg_name" with alg_name_order
table_std_left = table_std_left[alg_name_order]

table_std_right = pd.pivot_table(df, values='std_coverage', index='alpha', columns='alg_name')
# resort the column of the dataframe by matching "alg_name" with alg_name_order
table_std_right = table_std_right[alg_name_order]


max_values = table_left.max(axis=1)

# make the max value bold
#table = table.apply(lambda x: [f'\\textbf{{{val}}}' if val == row_max else val for val, row_max in zip(x, max_values)], axis=1)


# convert to latex table
latex_table_left = tabulate(table_left, headers='keys', tablefmt='latex')
latex_table_right = tabulate(table_right, headers='keys', tablefmt='latex')
latex_table_std_left = tabulate(table_std_left, headers='keys', tablefmt='latex')
latex_table_std_right = tabulate(table_std_right, headers='keys', tablefmt='latex')

# write to file
with open(dir+"latex_table_left.txt", "w") as text_file:
    text_file.write(latex_table_left)

with open(dir+"latex_table_right.txt", "w") as text_file:
    text_file.write(latex_table_right)

with open(dir+"latex_table_std_left.txt", "w") as text_file:
    text_file.write(latex_table_std_left)

with open(dir+"latex_table_std_right.txt", "w") as text_file:
    text_file.write(latex_table_std_right)