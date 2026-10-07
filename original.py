import pandas as pd
import ast
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# 1. LOAD DATA
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
data = movies.merge(credits, left_on="id", right_on="movie_id")

# 2. EXTRACT GENRE + DIRECTOR
def get_genre(x):
    try:
        genres = ast.literal_eval(x)
        return genres[0]["name"] if genres else "Unknown"
    except:
        return "Unknown"

def get_director(x):
    try:
        crew = ast.literal_eval(x)
        for member in crew:
            if member["job"] == "Director":
                return member["name"]
        return "Unknown"
    except:
        return "Unknown"

data["genre"] = data["genres"].apply(get_genre)
data["director"] = data["crew"].apply(get_director)

# 3. SELECT MOVIES
selected_movies = ["Inception", "Avatar", "Titanic", "The Dark Knight", "Interstellar"]
data = data[data["original_title"].isin(selected_movies)]

# 4. USER RATINGS
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

print("\n📊 Ratings Table:")
print(ratings)

# 5. USER-MOVIE MATRIX
matrix = ratings.pivot(index="user", columns="movie", values="rating").fillna(0)
print("\n📊 User-Movie Matrix:")
print(matrix)

# 6. USER SIMILARITY
similarity = cosine_similarity(matrix)
sim_df = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)
print("\n📊 Similarity Matrix:")
print(sim_df)

# 7. GRAPH 1: USER ↔ MOVIE
G1 = nx.Graph()
for _, row in ratings.iterrows():
    G1.add_edge(row["user"], row["movie"], weight=row["rating"])
plt.figure()
pos = nx.spring_layout(G1)
nx.draw(G1, pos, with_labels=True)
labels = nx.get_edge_attributes(G1, "weight")
nx.draw_networkx_edge_labels(G1, pos, edge_labels=labels)
plt.title("Graph 1: User-Movie Ratings")
plt.show()

# 8. GRAPH 2: FULL USER SIMILARITY NETWORK (different style)
G2 = nx.Graph()
for u1 in sim_df.index:
    for u2 in sim_df.columns:
        if u1 != u2:
            sim_score = round(sim_df.loc[u1, u2], 2)
            G2.add_edge(u1, u2, weight=sim_score)

print("\n📊 Pairwise Similarities:")
for u1 in sim_df.index:
    for u2 in sim_df.columns:
        if u1 != u2:
            print(f"{u1} ↔ {u2}: {sim_df.loc[u1, u2]:.3f}")

plt.figure(figsize=(8,6))
pos = nx.spring_layout(G2, seed=42)
nx.draw_networkx_nodes(G2, pos, node_size=1200, node_color="lightgreen")
nx.draw_networkx_labels(G2, pos, font_size=10, font_weight="bold")
nx.draw_networkx_edges(G2, pos)
labels = nx.get_edge_attributes(G2, "weight")
nx.draw_networkx_edge_labels(G2, pos, edge_labels=labels, font_size=8)
plt.title("Graph 2: Full User Similarity Network", fontsize=14)
plt.axis("off")
plt.show()

# 9. GRAPH 3: FULL KNOWLEDGE GRAPH
G3 = nx.Graph()
for _, row in ratings.iterrows():
    G3.add_edge(row["user"], row["movie"], relation="rated")
for _, row in data.iterrows():
    movie = row["original_title"]
    genre = row["genre"]
    director = row["director"]
    G3.add_edge(movie, genre, relation="genre")
    G3.add_edge(movie, director, relation="director")
for u1 in sim_df.index:
    for u2 in sim_df.columns:
        if u1 != u2:
            G3.add_edge(u1, u2, relation="similar")
plt.figure(figsize=(8,6))
pos = nx.spring_layout(G3)
nx.draw(G3, pos, with_labels=True, node_size=1500, font_size=8)
plt.title("Graph 3: Full Knowledge Graph")
plt.show()

# 10. RECOMMENDATION FUNCTION
def recommend(user):
    scores = {}
    for movie in matrix.columns:
        if matrix.loc[user, movie] == 0:
            num = 0
            den = 0
            for other in matrix.index:
                if other != user:
                    sim = sim_df.loc[user, other]
                    rating = matrix.loc[other, movie]
                    if rating > 0:
                        num += sim * rating
                        den += sim
            if den > 0:
                scores[movie] = round(num / den, 2)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

print("\n🎯 Recommendations for U3:")
print(recommend("U3"))

# 11. BONUS GRAPH (U3 FOCUS)
sim_df.loc["U3"].plot(kind="bar")
plt.title("User3 Similarity with Others")
plt.ylabel("Similarity Score")
plt.show()
