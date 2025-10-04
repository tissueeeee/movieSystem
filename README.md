# AI-Based Movie Recommendation System

## Overview

This is an AI-powered movie recommendation system built with Flask, leveraging content-based filtering to provide personalized movie suggestions. The system processes user preferences (e.g., favorite movies, genres, runtime, ratings) and uses natural language processing techniques to recommend similar movies from the TMDB 5000 Movies dataset.

Key AI components include:
- **TF-IDF Vectorization** for feature extraction from movie descriptions, genres, and keywords.
- **Cosine Similarity** for computing movie similarity scores.
- **Weighted Scoring** for ranking recommendations based on multiple criteria (genres, ratings, etc.).

The system supports user authentication, watchlists, search history logging, and an admin dashboard. Watchlists are user-specific and stored in a SQLite database to ensure privacy.

## Features

- **Personalized Recommendations**: AI-driven suggestions based on user input (favorite movie, genres, language, runtime, etc.).
- **User Authentication**: Secure login/register with hashed passwords.
- **User-Specific Watchlists**: Add/remove movies; each user sees only their own list.
- **Search History**: Logs user queries for analytics (admin view only).
- **Admin Dashboard**: View user stats, recent activity, and manage users.
- **TMDb API Integration**: Fetches details for movies not in the local dataset.
- **Frontend Enhancements**: Autocomplete for movie search, dynamic UI with JavaScript.
- **Responsive Design**: Modern CSS styling for desktop and mobile.

## Tech Stack

- **Backend**: Flask (Python 3.12+), SQLite (for user data/watchlists/history).
- **AI/ML**: scikit-learn (TF-IDF, cosine similarity), NumPy, Pandas.
- **Frontend**: HTML/Jinja2 templates, Vanilla JavaScript (app.js, watchlist.js).
- **External APIs**: The Movie Database (TMDB) API (key: `yourkey`).
- **Data**: TMDB 5000 Movies dataset (CSV files in `data/` folder).
- **Other**: Werkzeug (security), pytz (timezones, Malaysia UTC+8).

## Installation

### Prerequisites
- Python 3.12 or higher.
- pip (package manager).

### Steps
1. **Clone/Setup Project**:
   ```
   git clone <your-repo-url>
   cd ai-movie-recommender
   ```

2. **Create Virtual Environment**:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Create a `requirements.txt` file with:
   ```
   Flask>=3.0.0
   pandas>=2.2.2
   scikit-learn>=1.5.0
   numpy>=1.26.4
   requests>=2.31.0
   pytz>=2023.3
   Werkzeug>=3.0.0
   ```
   Then run:
   ```
   pip install -r requirements.txt
   ```

4. **Setup Database**:
   - The app auto-initializes `history.db` on first run (users, searches, watchlist, ratings tables).
   - Admin user is created automatically: `admin` / `admin123`.

5. **Add Dataset**:
   - Download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
   - Place them in a `data/` folder.

6. **Run the App**:
   ```
   python app.py
   ```
   - Access at `http://localhost:5000`.
   - Default admin: `admin` / `admin123`.

## Usage

1. **User Flow**:
   - Register/Login at `/register` or `/login`.
   - On homepage (`/`), select preferences (favorite movie, genres, etc.) and submit for recommendations.
   - Add movies to watchlist via the "+" button.
   - View personal watchlist at `/watchlist`.
   - Check search history at `/history`.

2. **Admin Flow**:
   - Login as admin.
   - Access dashboard at `/admin` for stats (users, searches, watchlists).
   - Manage users at `/admin_user` (view/delete).

3. **Recommendations**:
   - Input: Favorite movie (autocomplete), genres (multi-select), runtime, min rating, year range, etc.
   - Output: Up to 9 movies with match percentages (AI-scored).
   - Fallback: If movie not in dataset, fetches from TMDB API.

## API Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/` | GET | Homepage with recommendation form | Yes |
| `/recommend` | POST | Generate recommendations (JSON/form) | Yes |
| `/api/options` | GET | Available filters (genres, languages) | Yes |
| `/api/watchlist` | POST | Get watchlist movie details | Yes |
| `/api/watchlist/add` | POST | Add movie to user's watchlist | Yes |
| `/api/watchlist/remove` | POST | Remove movie from user's watchlist | Yes |
| `/api/watchlist/get` | GET | Get user's watchlist titles | Yes |
| `/login` | POST | User login | No |
| `/register` | POST | User registration | No |
| `/watchlist` | GET | Watchlist page | Yes |
| `/history` | GET | User's search history | Yes |
| `/admin` | GET | Admin dashboard | Admin only |
| `/admin_user` | GET | Manage users | Admin only |
| `/admin/delete_user/<username>` | POST | Delete user | Admin only |

## Database Schema (SQLite: `history.db`)

- **users**: `id`, `username` (unique), `email`, `password_hash`, `preferences` (JSON), `watched_movies` (JSON), `created_at`.
- **movies**: `id`, `title`, `genres` (JSON), `rating`, `description`, `features` (JSON).
- **ratings**: `user_id`, `movie_id`, `rating`, `timestamp`.
- **searches**: Logs user queries (`username`, `favorite_movie`, `genres`, etc., `timestamp`).
- **watchlist**: `id`, `username`, `movie_title`, `movie_id`, `added_at` (user-specific).

## Project Structure

```
ai-movie-recommender/
├── app.py                  # Main Flask app with AI logic
├── data/                   # Dataset CSVs
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── history.db              # SQLite database (auto-created)
├── static/                 # Frontend assets
│   ├── app.js              # JS for recommendations/autocomplete
│   └── watchlist.js        # JS for watchlist page
├── templates/              # Jinja2 HTML templates
│   ├── index.html          # Main recommendation page
│   ├── login.html
│   ├── register.html
│   ├── watchlist.html
│   ├── history.html
│   └── admin.html
└── requirements.txt        # Python dependencies
```
