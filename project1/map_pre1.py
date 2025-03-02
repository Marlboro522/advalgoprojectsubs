import osmnx as ox
import geopandas as gpd
import os
import json
import networkx as nx
from shapely.geometry import box
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial


def load_graph(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    G = ox.load_graphml(path)
    nodes, edges = ox.convert.graph_to_gdfs(G, nodes=True, edges=True)
    return G, nodes, edges


def find_bounding_box(G):
    north = max(G.nodes[node]["y"] for node in G.nodes)
    south = min(G.nodes[node]["y"] for node in G.nodes)
    east = max(G.nodes[node]["x"] for node in G.nodes)
    west = min(G.nodes[node]["x"] for node in G.nodes)
    return north, south, east, west


def create_square_grid(north, south, east, west, output_geojson, grid_size=10):
    if os.path.exists(output_geojson):
        print(f"File already exists: {output_geojson}")
        return
    lat_range = north - south
    lon_range = east - west
    step_size = min(lat_range, lon_range) / grid_size
    grid_cells = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell_west = west + j * step_size
            cell_east = cell_west + step_size
            cell_south = south + i * step_size
            cell_north = cell_south + step_size
            grid_cells.append(box(cell_west, cell_south, cell_east, cell_north))
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:4326")
    grid_gdf.to_file(output_geojson, driver="GeoJSON")
    print(f"Grid saved to {output_geojson}")
    return grid_gdf


def plot_grid_with_map(G, grid_gdf, output_file):
    if os.path.exists(output_file):
        print(f"File already exists: {output_file}")
        return
    fig, ax = plt.subplots(figsize=(10, 10))
    ox.plot_graph(
        G,
        ax=ax,
        node_size=0,
        edge_color="gray",
        edge_linewidth=0.5,
        show=False,
        close=False,
    )
    grid_gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=0.5, alpha=0.7)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Grid map saved to {output_file}")


def load_final_transit_nodes(output_dir="preprocessing_output"):
    with open(f"{output_dir}/final_transit_nodes.json", "r") as f:
        transit_nodes = set(map(int, json.load(f)))
    return transit_nodes


def assign_nodes_to_grid_and_find_boundaries(
    G, nodes, grid_gdf, output_dir="preprocessing_output"
):
    if os.path.exists(f"{output_dir}/node_to_grid.json"):
        print(f"File already exists: {output_dir}/node_to_grid.json")
        return
    os.makedirs(output_dir, exist_ok=True)
    node_to_grid = {}
    for node in G.nodes():
        node_lat, node_lon = nodes.loc[node, "y"], nodes.loc[node, "x"]
        for i, cell in grid_gdf.iterrows():
            if cell.geometry.contains(gpd.points_from_xy([node_lon], [node_lat])[0]):
                node_to_grid[node] = i
                break
    print(f"Assigned {len(node_to_grid)} nodes to grid cells")
    with open(f"{output_dir}/node_to_grid.json", "w") as f:
        json.dump(node_to_grid, f)
    print(f"Saved node-to-grid mapping: {output_dir}/node_to_grid.json")
    boundary_crossing_edges = []
    for u, v in G.edges():
        if node_to_grid.get(u) != node_to_grid.get(v):
            boundary_crossing_edges.append((u, v))
    print(
        f"Total edges: {len(G.edges())}, Found {len(boundary_crossing_edges)} boundary-crossing edges"
    )
    with open(f"{output_dir}/boundary_crossing_edges.json", "w") as f:
        json.dump(boundary_crossing_edges, f)
    print(f"Saved boundary-crossing edges: {output_dir}/boundary_crossing_edges.json")
    transit_nodes = set()
    for u, v in boundary_crossing_edges:
        transit_nodes.add(u)
        transit_nodes.add(v)
    print(
        f"Total transit nodes: {len(transit_nodes)}, Total boundary nodes: {len(boundary_crossing_edges)}"
    )
    with open(f"{output_dir}/final_transit_nodes.json", "w") as f:
        json.dump(list(transit_nodes), f)
    print(f"Saved transit nodes: {output_dir}/final_transit_nodes.json")


def load_boundary_crossing_nodes(output_dir="preprocessing_output"):
    with open(f"{output_dir}/final_transit_nodes.json", "r") as f:
        transit_nodes = set(map(int, json.load(f)))
    return transit_nodes


def _compute_partial_betweenness(G, node_subset):
    return nx.betweenness_centrality(G, weight="length", k=len(node_subset))


def compute_betweenness_for_transit_nodes(
    G,
    filename="preprocessing_output/transit_nodes_centrality.parquet",
    num_workers=None,
):
    if os.path.exists(filename):
        print(f"[INFO] Loading precomputed betweenness centrality from {filename}")
        return pd.read_parquet(filename).set_index("node").to_dict()["betweenness"]

    transit_nodes = load_boundary_crossing_nodes()
    print("[INFO] Computing betweenness centrality in parallel... This may take time.")

    nodes = list(G.nodes())
    num_workers = num_workers or mp.cpu_count()
    chunk_size = len(nodes) // num_workers
    node_chunks = [nodes[i : i + chunk_size] for i in range(0, len(nodes), chunk_size)]

    with mp.Pool(processes=num_workers) as pool:
        betweenness_results = pool.map(
            partial(_compute_partial_betweenness, G), node_chunks
        )

    centrality = {}
    for result in betweenness_results:
        centrality.update(result)

    filtered_centrality = {
        node: centrality[node] for node in transit_nodes if node in centrality
    }

    df = pd.DataFrame(
        list(filtered_centrality.items()), columns=["node", "betweenness"]
    )
    df.to_parquet(filename, index=False)
    print(f"[INFO] Transit node betweenness centrality saved to {filename}")

    return filtered_centrality


def select_top_transit_nodes(
    centrality_dict,
    percentile=65,
    filename="preprocessing_output/final_transit_nodes.json",
):
    values = list(centrality_dict.values())
    threshold = np.percentile(values, percentile)
    final_transit_nodes = {
        node for node, centrality in centrality_dict.items() if centrality >= threshold
    }
    print(
        f"[INFO] Selected {len(final_transit_nodes)} transit nodes (Top {percentile}%)"
    )
    with open(filename, "w") as f:
        json.dump(list(final_transit_nodes), f)
    print(f"Final transit nodes saved to {filename}")


def precompute_shortest_paths_graph_transit_nodes(
    G, filename="preprocessing_output/shortest_distances_from_transitnodes.parquet"
):
    if os.path.exists(filename):
        print(f"[INFO] Loading precomputed shortest path distances from {filename}")
        return pd.read_parquet(filename)
    with open("preprocessing_output/final_transit_nodes.json", "r") as f:
        transit_nodes = json.load(f)
        transit_nodes = set(map(int, transit_nodes))
    print(f"Loaded a total of {len(transit_nodes)} transit nodes")
    shortest_distances = {}
    for node in transit_nodes:
        shortest_distances[node] = nx.single_source_dijkstra_path_length(
            G, node, weight="length"
        )
    df = pd.DataFrame(shortest_distances).T
    df.to_parquet(filename)
    print(f"[INFO] Shortest path distances saved to {filename}")


def compute_distances_between_transit_nodes(filepath):
    with open(filepath, "r") as f:
        transit_nodes = json.load(f)
        transit_nodes = set(map(int, transit_nodes))
    shortest_distances = pd.read_parquet(
        "preprocessing_output/shortest_distances_from_transitnodes.parquet"
    )
    shortest_distances = shortest_distances.to_dict()
    distances = {}
    for node in transit_nodes:
        distances[node] = {}
        for other_node in transit_nodes:
            distances[node][other_node] = shortest_distances[node][other_node]
    with open("preprocessing_output/transit_node_distances.json", "w") as f:
        json.dump(distances, f)
    print(
        f"Saved transit node distances: preprocessing_output/transit_node_distances.json"
    )


def main():
    output_geojson = "resources/grid.geojson"
    output_img = "resources/grid_map.png"
    output_dir = "preprocessing_output"
    place = "resources/colorado_springs.graphml"
    G, nodes, edges = load_graph(place)
    north, south, east, west = find_bounding_box(G)
    print(f"Nodes: {len(nodes)} Edges: {len(edges)}")
    grid_gdf = create_square_grid(north, south, east, west, output_geojson)
    plot_grid_with_map(G, grid_gdf, output_img)
    assign_nodes_to_grid_and_find_boundaries(G, nodes, grid_gdf, output_dir)
    centrality_dict = compute_betweenness_for_transit_nodes(
        G, num_workers=16
    )

    select_top_transit_nodes(centrality_dict)
    print(f"You can think about multi processing now. ")
    precompute_shortest_paths_graph_transit_nodes(G)
    compute_distances_between_transit_nodes(
        filepath="preprocessing_output/final_transit_nodes.json"
    )


import time

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Preprocessing took {(time.time()-start)/60} minutes")
