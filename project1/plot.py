import os
import pandas as pd
import matplotlib.pyplot as plt

# Define file paths for different percentiles
percentile_files = {
    50: "preprocessing_output/tnr_vs_dijkstra_results_50.csv",
    65: "preprocessing_output/tnr_vs_dijkstra_results_65.csv",
    70: "preprocessing_output/tnr_vs_dijkstra_results_70.csv",
    75: "preprocessing_output/tnr_vs_dijkstra_results_75.csv",
    80: "preprocessing_output/tnr_vs_dijkstra_results_80.csv",
    85: "preprocessing_output/tnr_vs_dijkstra_results_85.csv",
    90: "preprocessing_output/tnr_vs_dijkstra_results_90.csv",
}

# Dictionary to store sanitized data for each percentile
sanitized_data = {}

# Load and filter data
for percentile, file_path in percentile_files.items():
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # Remove rows where the error percentage exceeds 15%
        df_filtered = df[df["Error (%)"] <= 25]

        # Store the filtered data
        sanitized_data[percentile] = df_filtered
        print(f"remaining {len(df_filtered)} rows")
    else:
        print(f"File not found: {file_path}")

# Dictionary to store analysis results
inflection_points = {}
avg_error_rates = {}
avg_dijkstra_times = {}
avg_tnr_times = {}

# Analyze each percentile
for percentile, df in sanitized_data.items():
    df_sorted = df.sort_values(by="Dijkstra Time (s)")  # Ensure data is sorted

    # Find the first occurrence where TNR time is less than Dijkstra time
    inflection_point = None
    for _, row in df_sorted.iterrows():
        if row["TNR Time (s)"] < row["Dijkstra Time (s)"]:
            inflection_point = row["Dijkstra Distance"]
            break

    inflection_points[percentile] = inflection_point
    avg_error_rates[percentile] = df["Error (%)"].mean()
    avg_dijkstra_times[percentile] = df["Dijkstra Time (s)"].mean()
    avg_tnr_times[percentile] = df["TNR Time (s)"].mean()

# Save results to CSV files
results_df = pd.DataFrame(
    {
        "Percentile": list(inflection_points.keys()),
        "Inflection Point (d_local)": list(inflection_points.values()),
        "Average Error (%)": list(avg_error_rates.values()),
        "Average Dijkstra Time (s)": list(avg_dijkstra_times.values()),
        "Average TNR Time (s)": list(avg_tnr_times.values()),
    }
)

results_df.to_csv("preprocessing_output/tnr_analysis_results.csv", index=False)

# --- Plot Inflection Points ---
plt.figure(figsize=(10, 6))
plt.plot(
    list(inflection_points.keys()),
    list(inflection_points.values()),
    marker="o",
    linestyle="-",
)
plt.xlabel("Transit Node Percentile")
plt.ylabel("Inflection Point (d_local) [meters]")
plt.title("inflecion point over percenteils")
plt.grid(True)
plt.xticks(list(inflection_points.keys()))
plt.savefig("plots/inflection_point_plot.png")
# plt.show()

# --- Plot Error Rates ---
plt.figure(figsize=(10, 6))
plt.plot(
    list(avg_error_rates.keys()),
    list(avg_error_rates.values()),
    marker="o",
    linestyle="-",
    color="red",
)
plt.xlabel("Transit Node Percentile")
plt.ylabel("Average Error (%)")
plt.title("avreege error over percentiles")
plt.grid(True)
plt.xticks(list(avg_error_rates.keys()))
plt.savefig("plots/error_rate_plot.png")
# plt.show()

# --- Plot Execution Time Comparison ---
plt.figure(figsize=(10, 6))
plt.plot(
    list(avg_dijkstra_times.keys()),
    list(avg_dijkstra_times.values()),
    marker="o",
    linestyle="-",
    label="Dijkstra Time",
    color="blue",
)
plt.plot(
    list(avg_tnr_times.keys()),
    list(avg_tnr_times.values()),
    marker="s",
    linestyle="-",
    label="TNR Time",
    color="green",
)
plt.xlabel("Transit Node Percentile")
plt.ylabel("Average Execution Time (s)")
plt.legend()
plt.grid(True)
plt.xticks(list(avg_dijkstra_times.keys()))
plt.savefig("plots/execution_time_plot.png")
# plt.show()

print(" comppltet, results and plots saved in `plots/`.")
