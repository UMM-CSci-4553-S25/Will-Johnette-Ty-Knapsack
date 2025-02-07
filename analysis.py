import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "run-2.csv")  # Replace with your actual CSV filename

# Load the CSV file
df = pd.read_csv(file_path)

# Get the maximum possible value for each seed
max_values_per_seed = df.groupby("seed")["max_value"].max()

# Apply performance transformation
def calculate_performance(row):
    if row["max_value"] < max_values_per_seed[row["seed"]]:
        return 100  # Worst possible performance
    return row["earliest_gen_of_max_value"]  # Otherwise, keep the generation

df["performance"] = df.apply(calculate_performance, axis=1)

# Pivot table to organize data for plotting
pivot_df = df.pivot(index="seed", columns=["cxpb", "mutpb"], values="performance")

# Plot
plt.figure(figsize=(12, 6))
for (cxpb, mutpb) in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[(cxpb, mutpb)], label=f"cxpb={cxpb}, mutpb={mutpb}")

plt.xlabel("Seed")
plt.ylabel("Performance (Lower is better)")
plt.title("Performance of cxpb/mutpb combinations across 30 seeds")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
plt.grid(True)
plt.show()
