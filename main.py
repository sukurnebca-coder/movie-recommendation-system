


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load datasets
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Analyze movie ratings
movie_stats = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()

# Add movie titles and genres
movie_stats = movie_stats.merge(movies, on="movieId")

# Display top 10 highly rated movies
# Consider movies with at least 50 ratings
popular_movies = movie_stats[
    movie_stats["rating_count"] >= 50
]

top_movies = popular_movies.sort_values(
    "average_rating", ascending=False
)

print("\nTOP 10 HIGHLY RATED MOVIES")
print(top_movies[
    ["title", "average_rating", "rating_count"]
].head(10).to_string(index=False))

# Display top 10 most rated movies
most_rated = movie_stats.sort_values(
    "rating_count", ascending=False
).head(10)

print("\nTOP 10 MOST POPULAR MOVIES")
print(most_rated[
    ["title", "average_rating", "rating_count"]
].to_string(index=False))

# Visualize popular movies
sns.barplot(
    data=most_rated,
    x="rating_count",
    y="title"
)

plt.title("Top 10 Most Rated Movies")
plt.xlabel("Number of Ratings")
plt.ylabel("Movie Title")
plt.tight_layout()
plt.show()

# STEP 12: PREPARE MOVIE FEATURES

from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

# Calculate average ratings for each movie
movie_stats = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()

# Combine ratings with movie details
movie_data = movies.merge(movie_stats, on="movieId", how="left")

# Fill missing ratings for movies that have no ratings
movie_data["average_rating"] = movie_data["average_rating"].fillna(0)
movie_data["rating_count"] = movie_data["rating_count"].fillna(0)

# Convert genres into a list
movie_data["genres_list"] = movie_data["genres"].apply(
    lambda x: x.split("|") if x != "(no genres listed)" else []
)

# Convert genres into numerical columns
mlb = MultiLabelBinarizer()
genre_features = mlb.fit_transform(movie_data["genres_list"])

genre_df = pd.DataFrame(
    genre_features,
    columns=mlb.classes_
)

# Scale average ratings
scaler = StandardScaler()
rating_feature = scaler.fit_transform(
    movie_data[["average_rating"]]
)

# Combine genres and ratings
features = np.concatenate(
    [genre_features, rating_feature],
    axis=1
)

# STEP 13: APPLY K-MEANS CLUSTERING

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

movie_data["cluster"] = kmeans.fit_predict(features)

# Display cluster information
print("\nMOVIE CLUSTERING RESULTS")

for cluster in sorted(movie_data["cluster"].unique()):
    print(f"\nCluster {cluster}:")
    print(
        movie_data[movie_data["cluster"] == cluster]["title"]
        .head(10)
        .to_string(index=False)
    )

# Save the clustered movie data
movie_data.to_csv("movie_clusters.csv", index=False)

# Visualize the number of movies in each cluster

# Visualize the number of movies in each cluster
cluster_counts = movie_data["cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
sns.barplot(
    x=cluster_counts.index,
    y=cluster_counts.values,
    color="steelblue"
)

plt.title("Number of Movies in Each Cluster")
plt.xlabel("Cluster Number")
plt.ylabel("Number of Movies")
plt.xticks(range(5), [str(i) for i in range(5)])
plt.tight_layout()
plt.savefig("cluster_visualization.png")
plt.close()


print("\nClustering completed successfully!")
print("Results saved to movie_clusters.csv")
print("Graph saved to cluster_visualization.png")

def recommend_movies(movie_name, number_of_recommendations=5):
    movies = pd.read_csv("movie_clusters.csv")

    # Find the movie entered by the user
    movie = movies[
        movies["title"].str.contains(
            movie_name, case=False, na=False
        )
    ]

    if movie.empty:
        print("Movie not found!")
        return

    # Select the first matching movie
    selected_movie = movie.iloc[0]

    # Get its cluster number
    cluster_id = selected_movie["cluster"]

    # Find other movies in the same cluster
    recommendations = movies[
        (movies["cluster"] == cluster_id) &
        (movies["movieId"] != selected_movie["movieId"])
    ].copy()

    # Sort by average rating, highest first
    recommendations = recommendations.sort_values(
        by=["average_rating", "rating_count"],
        ascending=[False, False]
    )

    # Display the recommendations
    print(f"\nMovies recommended for: {selected_movie['title']}")
    print("\nTop recommendations:")

    print(
        recommendations[
            ["title", "average_rating", "rating_count"]
        ].head(number_of_recommendations).to_string(index=False)
    )


    
    

# Rating and Genre Visualizations

import matplotlib.pyplot as plt
import pandas as pd

# Load the clustered movie data
movies = pd.read_csv("movie_clusters.csv")

# 1. Movie Average Rating Distribution
plt.figure(figsize=(8, 5))
plt.hist(movies["average_rating"].dropna(), bins=10, edgecolor="black")
plt.title("Distribution of Movie Ratings")
plt.xlabel("Average Rating")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.savefig("rating_distribution.png")
plt.show()
plt.close()

# 2. Movie Genre Distribution
genres = movies["genres"].dropna().str.split("|").explode()

# Remove movies without a listed genre
genres = genres[genres != "(no genres listed)"]

genre_counts = genres.value_counts().head(10)

plt.figure(figsize=(10, 5))
genre_counts.plot(kind="bar", edgecolor="black")
plt.title("Top 10 Most Common Movie Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("genre_distribution.png")
plt.show()
plt.close()

print("Rating and genre charts saved successfully!")


# Test the recommendation system
movie_name = input("Enter a movie name: ")
recommend_movies(movie_name)

