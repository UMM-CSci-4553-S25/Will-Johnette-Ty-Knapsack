import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "run-2.csv"  # Ensure this file is in the same directory as the script
df = pd.read_csv(file_path)

# Get the maximum possible value for each seed
max_values_per_seed = df.groupby("seed")["max_value"].max()

# Apply performance transformation
def calculate_performance(row):
    if row["max_value"] < max_values_per_seed[row["seed"]]:
        return 100  # Worst possible performance
    return row["earliest_gen_of_max_value"]  # Otherwise, keep the generation

df["performance"] = df.apply(calculate_performance, axis=1)

# Compute average performance per (cxpb, mutpb) combo
avg_performance = df.groupby(["cxpb", "mutpb"])["performance"].mean()

# Select top 5 performing combinations (lowest values)
top_5_combos = avg_performance.nsmallest(5).index

# Pivot table to organize data for plotting, filtering only top 5 combinations
pivot_df = df.pivot(index="seed", columns=["cxpb", "mutpb"], values="performance")
filtered_pivot_df = pivot_df[top_5_combos]

# Plot
plt.figure(figsize=(12, 6))
for (cxpb, mutpb) in filtered_pivot_df.columns:
    plt.plot(filtered_pivot_df.index, filtered_pivot_df[(cxpb, mutpb)], label=f"cxpb={cxpb}, mutpb={mutpb}")

plt.xlabel("Seed")
plt.ylabel("Performance (Lower is better)")
plt.title("Top 5 cxpb/mutpb Combinations Across 30 Seeds")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
plt.grid(True)
plt.show()
