import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --- SAMPLE RATINGS DATA ---
ratings = pd.DataFrame([
    ["U1", "Inception", 3],
    ["U1", "Avatar", 4],
    ["U1", "The Dark Knight", 5],

    ["U2", "Titanic", 3],
    ["U2", "Avatar", 3],
    ["U2", "The Dark Knight", 4],

    ["U3", "Interstellar", 3],
    ["U3", "Inception", 4],
    ["U3", "Avatar", 4],
    ["U3", "Titanic", 3],

    ["U4", "Avatar", 3],
    ["U4", "Titanic", 4],
    ["U4", "The Dark Knight", 5],

    ["U5", "Interstellar", 4]
], columns=["user", "movie", "rating"])

# --- USER-MOVIE MATRIX ---
matrix = ratings.pivot(index="user", columns="movie", values="rating").fillna(0)

# --- NORMALIZE RATINGS ---
matrix_norm = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0)

# --- USER SIMILARITY ---
similarity = cosine_similarity(matrix_norm)
sim_df = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

# --- PRINT SIMILARITY SCORES FOR USER 1 ---
print("\n Similarities for U1:")
for other in sim_df.index:
    if other != "U1":
        print(f"U1 ↔ {other}: {sim_df.loc['U1', other]:.3f}")

# --- BUILD SIMILARITY GRAPH CENTERED ON U1 ---
    target_user = "U1"
    G_sim = nx.DiGraph()

    for other in sim_df.index:
        if other != target_user:
            sim_score = sim_df.loc[target_user, other]
            # add every edge, no threshold
            G_sim.add_edge(target_user, other, weight=round(sim_score, 2))

# Layout: put target_user at center, others around circle
pos = nx.circular_layout(G_sim)
pos[target_user] = [0,0]  # force central node

plt.figure(figsize=(8,8))
nx.draw_networkx_nodes(G_sim, pos, node_size=1200, node_color="skyblue")
nx.draw_networkx_labels(G_sim, pos, font_size=10, font_weight="bold")
nx.draw_networkx_edges(G_sim, pos, arrows=True, arrowstyle="->", arrowsize=15)

# Edge labels = similarity scores
labels = nx.get_edge_attributes(G_sim, "weight")
nx.draw_networkx_edge_labels(G_sim, pos, edge_labels=labels, font_size=8)

plt.title(f"Similarity Network for {target_user}", fontsize=14)
plt.axis("off")
plt.show()
