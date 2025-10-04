AI-Based Movie Recommendation System
Overview
This is an AI-powered movie recommendation system built with Flask, leveraging content-based filtering to provide personalized movie suggestions. The system processes user preferences (e.g., favorite movies, genres, runtime, ratings) and uses natural language processing techniques to recommend similar movies from the TMDB 5000 Movies dataset.
Key AI components include:

TF-IDF Vectorization for feature extraction from movie descriptions, genres, and keywords.
Cosine Similarity for computing movie similarity scores.
Weighted Scoring for ranking recommendations based on multiple criteria (genres, ratings, etc.).

The system supports user authentication, watchlists, search history logging, and an admin dashboard. Watchlists are user-specific and stored in a SQLite database to ensure privacy.
Features

Personalized Recommendations: AI-driven suggestions based on user input (favorite movie, genres, language, runtime, etc.).
User Authentication: Secure login/register with hashed passwords.
User-Specific Watchlists: Add/remove movies; each user sees only their own list.
Search History: Logs user queries for analytics (admin view only).
Admin Dashboard: View user stats, recent activity, and manage users.
TMDb API Integration: Fetches details for movies not in the local dataset.
Frontend Enhancements: Autocomplete for movie search, dynamic UI with JavaScript.
Responsive Design: Modern CSS styling for desktop and mobile.

Tech Stack

Backend: Flask (Python 3.12+), SQLite (for user data/watchlists/history).
AI/ML: scikit-learn (TF-IDF, cosine similarity), NumPy, Pandas.
Frontend: HTML/Jinja2 templates, Vanilla JavaScript (app.js, watchlist.js).
External APIs: The Movie Database (TMDB) API (key: 44f180dccfbc6d2d9ed74bdd398cf242).
Data: TMDB 5000 Movies dataset (CSV files in data/ folder).
Other: Werkzeug (security), pytz (timezones, Malaysia UTC+8).

Installation
Prerequisites

Python 3.12 or higher.
pip (package manager).

<b>Usage</b>

User Flow:

Register/Login at /register or /login.
On homepage (/), select preferences (favorite movie, genres, etc.) and submit for recommendations.
Add movies to watchlist via the "+" button.
View personal watchlist at /watchlist.
Check search history at /history.


Admin Flow:

Login as admin.
Access dashboard at /admin for stats (users, searches, watchlists).
Manage users at /admin_user (view/delete).


Recommendations:

Input: Favorite movie (autocomplete), genres (multi-select), runtime, min rating, year range, etc.
Output: Up to 9 movies with match percentages (AI-scored).
Fallback: If movie not in dataset, fetches from TMDB API.
