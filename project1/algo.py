import networkx as nx
import pandas as pd
import json
import time
import numpy as np
import os
import random

G = nx.read_graphml("resources/colorado_springs.graphml")

for u, v, data in G.edges(data=True):
    if "length" in data:
        data["length"] = float(data["length"])

SHORTEST_DISTANCES_PATH = (
    "preprocessing_output/shortest_distances_from_transitnodes.parquet"
)
shortest_distances_df = pd.read_parquet(SHORTEST_DISTANCES_PATH)

shortest_distances = {
    str(k): {str(inner_k): float(v) for inner_k, v in v.items()}
    for k, v in shortest_distances_df.astype(str).to_dict().items()
}

with open("preprocessing_output/final_transit_nodes.json", "r") as f:
    transit_nodes = set(json.load(f))

with open("preprocessing_output/transit_node_distances.json", "r") as f:
    transit_node_distances = json.load(f)

nearest_transit_node = {}
for i in shortest_distances.keys():
    filtered_distances = {k: v for k, v in shortest_distances[i].items() if k != i}

    if filtered_distances:
        nearest_transit_node[i] = min(
            filtered_distances.items(), key=lambda x: float(x[1])
        )
    else:
        nearest_transit_node[i] = (None, float("inf"))


def dijkstra_shortest_path(source, target):
    try:
        return nx.shortest_path_length(G, source, target, weight="length")
    except nx.NetworkXNoPath:
        return float("inf")


def tnr_shortest_path(source, target, d_local=1000):
    source = str(source)
    target = str(target)

    if source not in G.nodes or target not in G.nodes:
        return float("inf")

    if source in shortest_distances and target in shortest_distances[source]:
        local_distance = shortest_distances[source][target]
        if local_distance <= d_local:
            return local_distance

    nearest_source_tn, dist_source_to_tn = nearest_transit_node.get(
        source, (None, float("inf"))
    )
    nearest_target_tn, dist_target_to_tn = nearest_transit_node.get(
        target, (None, float("inf"))
    )

    if nearest_source_tn is None or nearest_target_tn is None:
        return float("inf")

    dist_tn_to_tn = transit_node_distances.get(nearest_source_tn, {}).get(
        nearest_target_tn, float("inf")
    )

    return dist_source_to_tn + dist_tn_to_tn + dist_target_to_tn


num_tests = 100
all_nodes = list(G.nodes)
test_cases = [
    (random.choice(all_nodes), random.choice(all_nodes)) for _ in range(num_tests)
]

d_local = 1000

results = []

for source, target in test_cases:
    start_time = time.time()
    dijkstra_distance = dijkstra_shortest_path(source, target)
    dijkstra_time = time.time() - start_time

    start_time = time.time()
    tnr_distance = tnr_shortest_path(source, target, d_local)
    tnr_time = time.time() - start_time

    if dijkstra_distance == float("inf") or tnr_distance == float("inf"):
        continue

    error_percentage = (
        abs(dijkstra_distance - tnr_distance) / max(dijkstra_distance, 1) * 100
    )

    results.append(
        (
            source,
            target,
            dijkstra_distance,
            tnr_distance,
            dijkstra_time,
            tnr_time,
            error_percentage,
        )
    )

df_results = pd.DataFrame(
    results,
    columns=[
        "Source",
        "Target",
        "Dijkstra Distance",
        "TNR Distance",
        "Dijkstra Time (s)",
        "TNR Time (s)",
        "Error (%)",
    ],
)

print(df_results.head())

df_results.to_csv("preprocessing_output/tnr_vs_dijkstra_results.csv", index=False)

print(
    "Testing complete! Results saved to `preprocessing_output/tnr_vs_dijkstra_results.csv`."
)
