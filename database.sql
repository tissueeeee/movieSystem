-- Users table
CREATE TABLE users (
    id INT PRIMARY KEY,
    preferences JSON,
    watched_movies JSON
);

-- Movies table  
CREATE TABLE movies (
    id INT PRIMARY KEY,
    title VARCHAR(255),
    genres JSON,
    rating DECIMAL,
    description TEXT,
    features JSON
);

-- Ratings table
CREATE TABLE ratings (
    user_id INT,
    movie_id INT,
    rating DECIMAL,
    timestamp DATETIME
);