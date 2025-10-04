import pandas as pd

file_path = "data/tmdb_5000_movies.csv"
df = pd.read_csv(file_path)

unique_languages = df['original_language'].unique()

print(unique_languages.tolist())
