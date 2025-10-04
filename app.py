import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import logging
import requests
import sqlite3
from datetime import datetime
import pytz
import ast
import re
import hashlib
from werkzeug.security import generate_password_hash, check_password_hash

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6'

# TMDb API setup
TMDB_API_KEY = '44f180dccfbc6d2d9ed74bdd398cf242'
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

# Set Malaysia time zone (UTC+08:00)
malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')

LANGUAGE_MAP = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'cn': 'Chinese',
    'ru': 'Russian',
    'pt': 'Portuguese',
    'hi': 'Hindi',
    'ar': 'Arabic',
    'tr': 'Turkish',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'no': 'Norwegian',
    'da': 'Danish',
    'fi': 'Finnish',
    'pl': 'Polish',
    'cs': 'Czech',
    'el': 'Greek',
    'th': 'Thai',
    'id': 'Indonesian',
    'ms': 'Malay',
    'vi': 'Vietnamese',
    'he': 'Hebrew',
    'uk': 'Ukrainian',
    'hu': 'Hungarian',
    'ro': 'Romanian',
    'bg': 'Bulgarian',
    'fa': 'Persian',
    'ta': 'Tamil',
    'te': 'Telugu',
    'bn': 'Bengali',
    'sr': 'Serbian',
    'hr': 'Croatian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'et': 'Estonian',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'ka': 'Georgian',
    'az': 'Azerbaijani',
    'uz': 'Uzbek',
    'kk': 'Kazakh',
    'af': 'Afrikaans',
    'sw': 'Swahili',
    'zu': 'Zulu',
    'xh': 'Xhosa',
    'st': 'Southern Sotho',
    'tn': 'Tswana',
    'ts': 'Tsonga',
    'ss': 'Swati',
    've': 'Venda',
    'nr': 'Southern Ndebele',
    'nd': 'Northern Ndebele',
    'rw': 'Kinyarwanda',
    'so': 'Somali',
    'am': 'Amharic',
    'om': 'Oromo',
    'ti': 'Tigrinya',
    'yo': 'Yoruba',
    'ig': 'Igbo',
    'ha': 'Hausa',
    'my': 'Burmese',
    'km': 'Khmer',
    'lo': 'Lao',
    'mn': 'Mongolian',
    'ne': 'Nepali',
    'si': 'Sinhala',
    'pa': 'Punjabi',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'or': 'Odia',
    'sa': 'Sanskrit',
    'ur': 'Urdu',
    'ps': 'Pashto',
    'sd': 'Sindhi',
    'tk': 'Turkmen',
    'ky': 'Kyrgyz',
    'tg': 'Tajik',
    'uz': 'Uzbek',
    'kk': 'Kazakh',
    'hy': 'Armenian',
    'ka': 'Georgian',
    'az': 'Azerbaijani',
    'be': 'Belarusian',
    'mo': 'Moldovan',
    'sq': 'Albanian',
    'bs': 'Bosnian',
    'mk': 'Macedonian',
    'mt': 'Maltese',
    'ga': 'Irish',
    'cy': 'Welsh',
    'gd': 'Scottish Gaelic',
    'br': 'Breton',
    'co': 'Corsican',
    'eu': 'Basque',
    'gl': 'Galician',
    'ca': 'Catalan',
    'lb': 'Luxembourgish',
    'is': 'Icelandic',
    'fo': 'Faroese',
    'yi': 'Yiddish',
    'jv': 'Javanese',
    'su': 'Sundanese',
    'fil': 'Filipino',
    'tl': 'Tagalog',
    'ceb': 'Cebuano',
    'ilo': 'Ilocano',
    'war': 'Waray',
    'pam': 'Pampanga',
    'bik': 'Bikol',
    'hil': 'Hiligaynon',
    'mag': 'Magahi',
    'mai': 'Maithili',
    'bh': 'Bihari',
    'as': 'Assamese',
    'mni': 'Manipuri',
    'kok': 'Konkani',
    'doi': 'Dogri',
    'ks': 'Kashmiri',
    'sat': 'Santali',
    'sd': 'Sindhi',
    'ur': 'Urdu',
    'bo': 'Tibetan',
    'dz': 'Dzongkha',
    'my': 'Burmese',
    'km': 'Khmer',
    'lo': 'Lao',
    'vi': 'Vietnamese',
    'th': 'Thai',
    'ms': 'Malay',
    'id': 'Indonesian',
    'jw': 'Javanese',
    'su': 'Sundanese',
    'tl': 'Tagalog',
    'fil': 'Filipino',
    'ceb': 'Cebuano',
    'ilo': 'Ilocano',
    'war': 'Waray',
    'pam': 'Pampanga',
    'bik': 'Bikol',
    'hil': 'Hiligaynon',
    'mag': 'Magahi',
    'mai': 'Maithili',
    'bh': 'Bihari',
    'as': 'Assamese',
    'mni': 'Manipuri',
    'kok': 'Konkani',
    'doi': 'Dogri',
    'ks': 'Kashmiri',
    'sat': 'Santali',
    'sd': 'Sindhi',
    'ur': 'Urdu',
    'bo': 'Tibetan',
    'dz': 'Dzongkha',
}
def is_admin():
        """Check if current user is admin"""
        return session.get('username') == 'admin'
class MovieRecommender:
    def __init__(self):
        self.df = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.load_and_preprocess_data()
        self.initialize_database()
        db_path = 'history.db'
        logger.info(f"Attempting to initialize database at {db_path}")
        self.initialize_database()
        logger.info(f"Database initialization completed for {db_path}")
        self.update_database_schema()
        logger.info(f"Database update completed for {db_path}")
        self.create_admin_user()
        logger.info(f"Admin create completed for {db_path}")

    def initialize_database(self):
        try:
            conn = sqlite3.connect('history.db')
            cursor = conn.cursor()
            cursor.executescript('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    preferences JSON,
                    watched_movies JSON
                );
                CREATE TABLE IF NOT EXISTS movies (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    genres JSON,
                    rating DECIMAL,
                    description TEXT,
                    features JSON
                );
                CREATE TABLE IF NOT EXISTS ratings (
                    user_id INTEGER,
                    movie_id INTEGER,
                    rating DECIMAL,
                    timestamp DATETIME
                );
                CREATE TABLE IF NOT EXISTS searches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    favorite_movie TEXT,
                    genres TEXT,
                    original_language TEXT,
                    min_rating DECIMAL,
                    max_runtime INTEGER,
                    min_runtime INTEGER,
                    keywords TEXT,
                    production_companies TEXT,
                    release_year_start INTEGER,
                    release_year_end INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    movie_title TEXT NOT NULL,
                    movie_id INTEGER,
                    added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (username) REFERENCES users(username)
                );
            ''')
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
                        
    def update_database_schema(self):
        """Update database schema to include email and password fields"""
        try:
            conn = sqlite3.connect('history.db')
            cursor = conn.cursor()
            
            # Check if email column exists
            cursor.execute("PRAGMA table_info(users)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'email' not in columns:
                cursor.execute('ALTER TABLE users ADD COLUMN email TEXT')
                logger.info("Added email column to users table")
            
            if 'password_hash' not in columns:
                cursor.execute('ALTER TABLE users ADD COLUMN password_hash TEXT')
                logger.info("Added password_hash column to users table")
            
            if 'created_at' not in columns:
                cursor.execute('ALTER TABLE users ADD COLUMN created_at DATETIME')
                logger.info("Added created_at column to users table")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating database schema: {e}")

    def create_user(self, username, email, password):
        """Create a new user account"""
        try:
            conn = sqlite3.connect('history.db')
            cursor = conn.cursor()
            
            # Check if username already exists
            cursor.execute('SELECT username FROM users WHERE username = ?', (username,))
            if cursor.fetchone():
                conn.close()
                return {'success': False, 'error': 'Username already exists'}
            
            # Check if email already exists
            cursor.execute('SELECT username FROM users WHERE email = ?', (email,))
            if cursor.fetchone():
                conn.close()
                return {'success': False, 'error': 'Email already registered'}
            
            # Hash the password
            password_hash = generate_password_hash(password)
            
            # Insert new user
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, preferences, watched_movies, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (username, email, password_hash, '{}', '[]', datetime.now(malaysia_tz).strftime('%Y-%m-%d %H:%M:%S')))
            
            conn.commit()
            conn.close()
            
            logger.info(f"New user created: {username}")
            return {'success': True, 'message': 'Account created successfully'}
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return {'success': False, 'error': 'Failed to create account'}

    def authenticate_user(self, username, password):
        """Authenticate user login"""
        try:
            conn = sqlite3.connect('history.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
            result = cursor.fetchone()
            conn.close()
            
            if result and check_password_hash(result[0], password):
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return False
        
    def create_admin_user(self):
        """Create admin user if it doesn't exist"""
        try:
            conn = sqlite3.connect('history.db')
            cursor = conn.cursor()
            
            # Check if admin user exists
            cursor.execute('SELECT username FROM users WHERE username = ?', ('admin',))
            if not cursor.fetchone():
                # Create admin user
                password_hash = generate_password_hash('admin123')
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash, preferences, watched_movies, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', ('admin', 'admin@justmovie.com', password_hash, '{}', '[]', datetime.now(malaysia_tz).strftime('%Y-%m-%d %H:%M:%S')))
                
                conn.commit()
                logger.info("Admin user created successfully")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error creating admin user: {e}")

    def load_and_preprocess_data(self):
        try:
            logger.debug("Loading dataset from data/tmdb_5000_movies.csv")
            self.df = pd.read_csv('data/tmdb_5000_movies.csv')
            
            # Parse JSON fields safely
            self.df['genres_list'] = self.df['genres'].apply(self.parse_json_field)
            self.df['keywords_list'] = self.df['keywords'].apply(self.parse_json_field)
            self.df['production_companies_list'] = self.df['production_companies'].apply(self.parse_json_field)
            self.df['production_countries_list'] = self.df['production_countries'].apply(self.parse_json_field)
            self.df['spoken_languages_list'] = self.df['spoken_languages'].apply(self.parse_json_field)
            
            # Convert genres to readable format
            self.df['genres_str'] = self.df['genres_list'].apply(lambda x: ', '.join(x) if x else '')
            self.df['keywords_str'] = self.df['keywords_list'].apply(lambda x: ', '.join(x) if x else '')
            self.df['production_companies_str'] = self.df['production_companies_list'].apply(lambda x: ', '.join(x) if x else '')
            
            # Process release date
            self.df['release_date'] = pd.to_datetime(self.df['release_date'], errors='coerce')
            self.df['release_year'] = self.df['release_date'].dt.year
            self.df['release_decade'] = (self.df['release_year'] // 10) * 10
            
            # Handle runtime
            self.df['runtime'] = pd.to_numeric(self.df['runtime'], errors='coerce')
            self.df['runtime'] = self.df['runtime'].fillna(self.df['runtime'].median())
            
            # Clean and prepare data
            self.df = self.df.dropna(subset=['title', 'overview', 'vote_average', 'release_date'])
            self.df = self.df[self.df['vote_average'] > 0]  # Remove movies with no ratings
            
            # Create enhanced content for similarity calculation
            self.df['enhanced_content'] = (
                self.df['genres_str'].fillna('') + ' ' +
                self.df['keywords_str'].fillna('') + ' ' +
                self.df['overview'].fillna('') + ' ' +
                self.df['tagline'].fillna('')
            )
            
            # Initialize TF-IDF
            self.tfidf = TfidfVectorizer(
                stop_words='english', 
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            self.tfidf_matrix = self.tfidf.fit_transform(self.df['enhanced_content'])
            self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            
            logger.info(f"Dataset loaded with {len(self.df)} movies")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            self.create_dummy_data()

    def parse_json_field(self, field):
        """Parse JSON fields from the dataset"""
        try:
            if pd.isna(field) or field == '':
                return []
            
            # Handle string representation of lists
            if isinstance(field, str):
                # Replace single quotes with double quotes for proper JSON
                field = field.replace("'", '"')
                # Handle None values
                field = field.replace('None', 'null')
                
            data = json.loads(field)
            
            if isinstance(data, list):
                # Extract names from the JSON objects
                if data and isinstance(data[0], dict):
                    if 'name' in data[0]:
                        return [item['name'] for item in data if 'name' in item]
                    elif 'iso_639_1' in data[0]:  # For language codes
                        return [item['name'] for item in data if 'name' in item]
                    elif 'iso_3166_1' in data[0]:  # For country codes
                        return [item['name'] for item in data if 'name' in item]
            
            return []
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            return []

    def create_dummy_data(self):
        """Create dummy data if the main dataset fails to load"""
        dummy_data = {
            'id': [1, 2, 3, 4, 5],
            'title': ['The Matrix', 'Avatar', 'Inception', 'The Dark Knight', 'Pulp Fiction'],
            'genres_str': ['Action, Sci-Fi', 'Action, Adventure, Fantasy, Sci-Fi', 'Action, Drama, Sci-Fi',
                          'Action, Crime, Drama', 'Crime, Drama'],
            'runtime': [136, 162, 148, 152, 154],
            'vote_average': [8.7, 7.9, 8.8, 9.0, 8.9],
            'release_date': pd.to_datetime(['1999-03-31', '2009-12-18', '2010-07-16', '2008-07-18', '1994-10-14']),
            'release_year': [1999, 2009, 2010, 2008, 1994],
            'overview': ['A hacker discovers reality is a simulation', 'Humans colonize an alien planet',
                        'Dream within a dream heist', 'Batman fights the Joker', 'Interconnected crime stories'],
            'original_language': ['en', 'en', 'en', 'en', 'en'],
            'keywords_str': ['virtual reality, simulation', 'alien, planet, future', 'dreams, heist', 'batman, joker', 'crime, stories'],
            'production_companies_str': ['Warner Bros', 'Twentieth Century Fox', 'Warner Bros', 'Warner Bros', 'Miramax'],
            'enhanced_content': ['Action Sci-Fi virtual reality A hacker discovers reality', 
                               'Action Adventure Fantasy Sci-Fi alien planet Humans colonize',
                               'Action Drama Sci-Fi dreams heist Dream within a dream', 
                               'Action Crime Drama batman joker Batman fights',
                               'Crime Drama crime stories Interconnected crime stories']
        }
        
        self.df = pd.DataFrame(dummy_data)
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=100)
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['enhanced_content'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def get_recommendations(self, user_preferences=None, is_initial=False):
        try:
            if not user_preferences:
                user_preferences = {
                    'favorite_movie': '', 'genres': '', 'original_language': '',
                    'min_rating': 0, 'max_runtime': None, 'min_runtime': None,
                    'keywords': '', 'production_companies': '',
                    'release_year_start': None, 'release_year_end': None
                }

            df_filtered = self.df.copy()
            
            # Apply filters
            if user_preferences.get('genres'):
                genre_list = [g.strip() for g in user_preferences['genres'].split(',') if g.strip()]
                if genre_list:
                    genre_mask = df_filtered['genres_str'].str.contains('|'.join(genre_list), case=False, na=False)
                    df_filtered = df_filtered[genre_mask]
            
            if user_preferences.get('original_language'):
                df_filtered = df_filtered[df_filtered['original_language'] == user_preferences['original_language']]
            
            if user_preferences.get('min_rating'):
                df_filtered = df_filtered[df_filtered['vote_average'] >= float(user_preferences['min_rating'])]
            
            max_runtime = user_preferences.get('max_runtime')
            min_runtime = user_preferences.get('min_runtime')
            # Fix for max_runtime
            if max_runtime:
                try:
                    if isinstance(max_runtime, str) and max_runtime.strip():
                        max_runtime = int(max_runtime)
                        df_filtered = df_filtered[df_filtered['runtime'] <= max_runtime]
                    elif isinstance(max_runtime, (int, float)):
                        df_filtered = df_filtered[df_filtered['runtime'] <= max_runtime]
                except (ValueError, TypeError):
                    pass  # Skip filtering if conversion fails

            # Fix for min_runtime
            if min_runtime:
                try:
                    if isinstance(min_runtime, str) and min_runtime.strip():
                        min_runtime = int(min_runtime)
                        df_filtered = df_filtered[df_filtered['runtime'] >= min_runtime]
                    elif isinstance(min_runtime, (int, float)):
                        df_filtered = df_filtered[df_filtered['runtime'] >= min_runtime]
                except (ValueError, TypeError):
                    pass  # Skip filtering if conversion fails
            
            if user_preferences.get('keywords'):
                keyword_list = [k.strip() for k in user_preferences['keywords'].split(',') if k.strip()]
                if keyword_list:
                    keyword_mask = df_filtered['keywords_str'].str.contains('|'.join(keyword_list), case=False, na=False)
                    df_filtered = df_filtered[keyword_mask]
            
            if user_preferences.get('production_companies'):
                company_list = [c.strip() for c in user_preferences['production_companies'].split(',') if c.strip()]
                if company_list:
                    company_mask = df_filtered['production_companies_str'].str.contains('|'.join(company_list), case=False, na=False)
                    df_filtered = df_filtered[company_mask]
            
            release_year_start = user_preferences.get('release_year_start')
            release_year_end = user_preferences.get('release_year_end')
            # Fix for release_year_start
            if release_year_start:
                try:
                    if isinstance(release_year_start, str) and release_year_start.strip():
                        release_year_start = int(release_year_start)
                        df_filtered = df_filtered[df_filtered['release_year'] >= release_year_start]
                    elif isinstance(release_year_start, (int, float)):
                        df_filtered = df_filtered[df_filtered['release_year'] >= release_year_start]
                except (ValueError, TypeError):
                    pass  # Skip filtering if conversion fails

            # Fix for release_year_end
            if release_year_end:
                try:
                    if isinstance(release_year_end, str) and release_year_end.strip():
                        release_year_end = int(release_year_end)
                        df_filtered = df_filtered[df_filtered['release_year'] <= release_year_end]
                    elif isinstance(release_year_end, (int, float)):
                        df_filtered = df_filtered[df_filtered['release_year'] <= release_year_end]
                except (ValueError, TypeError):
                    pass  # Skip filtering if conversion fails
            
            # Handle favorite movie similarity
            if user_preferences.get('favorite_movie'):
                favorite_movie = user_preferences['favorite_movie']
                
                # Try to find the movie in the dataset first
                movie_matches = self.df[self.df['title'].str.contains(favorite_movie, case=False, na=False)]
                
                if not movie_matches.empty:
                    movie_idx = movie_matches.index[0]
                    sim_scores = list(enumerate(self.cosine_sim[movie_idx]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                    
                    # Get indices of similar movies within our filtered dataset
                    similar_indices = []
                    for idx, score in sim_scores[1:101]:  # Skip the movie itself, get top 100
                        if idx in df_filtered.index:
                            similar_indices.append(idx)
                            if len(similar_indices) >= 50:
                                break
                    
                    if similar_indices:
                        recommendations = df_filtered.loc[similar_indices]
                    else:
                        recommendations = df_filtered.sample(n=min(20, len(df_filtered)))
                else:
                    # Try to fetch from TMDb if not found
                    tmdb_movie = self.fetch_tmdb_movie(favorite_movie)
                    if tmdb_movie:
                        # Use content-based similarity with TMDb data
                        tmdb_content = f"{tmdb_movie['genres']} {tmdb_movie['overview']}"
                        tmdb_tfidf = self.tfidf.transform([tmdb_content])
                        similarities = cosine_similarity(tmdb_tfidf, self.tfidf_matrix[df_filtered.index]).flatten()
                        
                        # Get top similar movies
                        similar_indices = similarities.argsort()[::-1][:20]
                        recommendations = df_filtered.iloc[similar_indices]
                    else:
                        recommendations = df_filtered.sample(n=min(20, len(df_filtered)))
            else:
                # No favorite movie, use filtered results
                recommendations = df_filtered.sample(n=min(50, len(df_filtered)))
            
            # Calculate sophisticated matching scores
            recommendations = self.calculate_matching_scores(recommendations, user_preferences)
            
            # Sort by final score and get top recommendations
            recommendations = recommendations.sort_values('final_score', ascending=False).head(9)
            
            # Prepare result
            result = {
                "recommendations": [{
                    'id': int(row['id']) if pd.notna(row['id']) else 0,
                    'title': row['title'],
                    'year': int(row['release_year']) if pd.notna(row['release_year']) else 'N/A',
                    'runtime': int(row['runtime']) if pd.notna(row['runtime']) else 'N/A',
                    'rating': float(row['vote_average']) if pd.notna(row['vote_average']) else 0.0,
                    'match_percentage': float(row['final_score']),
                    'description': row['overview'] if pd.notna(row['overview']) else 'No description available',
                    'genres': row['genres_str'] if pd.notna(row['genres_str']) else 'N/A',
                    'language': row['original_language'] if pd.notna(row['original_language']) else 'N/A',
                    'keywords': row['keywords_str'] if pd.notna(row['keywords_str']) else 'N/A',
                    'production_companies': row['production_companies_str'] if pd.notna(row['production_companies_str']) else 'N/A',
                    'watch_link': f'https://www.themoviedb.org/movie/{int(row["id"])}' if pd.notna(row['id']) else 'https://www.themoviedb.org'
                } for _, row in recommendations.iterrows()],
                "total_filtered": len(df_filtered),
                "selected_movie": user_preferences.get('favorite_movie') if user_preferences.get('favorite_movie') else None
            }
            
            # Log search history
            if 'username' in session and result['recommendations'] and not is_initial:
                self.log_search_history(session['username'], user_preferences)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in recommendations: {e}")
            return {"error": str(e)}

    def calculate_matching_scores(self, recommendations, user_preferences):
        """Calculate sophisticated matching scores based on multiple criteria"""
        
        # Initialize scores
        recommendations = recommendations.copy()
        recommendations['genre_score'] = 0.0
        recommendations['language_score'] = 0.0
        recommendations['rating_score'] = 0.0
        recommendations['runtime_score'] = 0.0
        recommendations['keyword_score'] = 0.0
        recommendations['company_score'] = 0.0
        recommendations['year_score'] = 0.0
        recommendations['popularity_score'] = 0.0
        
        # Genre matching (30% weight)
        if user_preferences.get('genres'):
            preferred_genres = [g.strip().lower() for g in user_preferences['genres'].split(',') if g.strip()]
            recommendations['genre_score'] = recommendations['genres_str'].apply(
                lambda x: self.calculate_genre_match(x, preferred_genres) if pd.notna(x) else 0.0
            )
        
        # Language matching (10% weight)
        if user_preferences.get('original_language'):
            recommendations['language_score'] = recommendations['original_language'].apply(
                lambda x: 1.0 if x == user_preferences['original_language'] else 0.0
            )
        
        # Rating boost (20% weight)
        min_rating = float(user_preferences.get('min_rating', 0))
        recommendations['rating_score'] = recommendations['vote_average'].apply(
            lambda x: min((x - min_rating) / (10 - min_rating), 1.0) if x >= min_rating else 0.0
        )
        
        # Runtime preference (10% weight)
        if user_preferences.get('max_runtime') or user_preferences.get('min_runtime'):
            recommendations['runtime_score'] = recommendations['runtime'].apply(
                lambda x: self.calculate_runtime_score(x, user_preferences)
            )
        
        # Keyword matching (15% weight)
        if user_preferences.get('keywords'):
            preferred_keywords = [k.strip().lower() for k in user_preferences['keywords'].split(',') if k.strip()]
            recommendations['keyword_score'] = recommendations['keywords_str'].apply(
                lambda x: self.calculate_keyword_match(x, preferred_keywords) if pd.notna(x) else 0.0
            )
        
        # Production company matching (5% weight)
        if user_preferences.get('production_companies'):
            preferred_companies = [c.strip().lower() for c in user_preferences['production_companies'].split(',') if c.strip()]
            recommendations['company_score'] = recommendations['production_companies_str'].apply(
                lambda x: self.calculate_company_match(x, preferred_companies) if pd.notna(x) else 0.0
            )
        
        # Year range matching (10% weight)
        if user_preferences.get('release_year_start') or user_preferences.get('release_year_end'):
            recommendations['year_score'] = recommendations['release_year'].apply(
                lambda x: self.calculate_year_score(x, user_preferences)
            )
        
        # Calculate final weighted score
        weights = {
            'genre_score': 0.30,
            'rating_score': 0.20,
            'keyword_score': 0.15,
            'language_score': 0.10,
            'runtime_score': 0.10,
            'year_score': 0.10,
            'company_score': 0.05
        }
        
        recommendations['final_score'] = sum(
            recommendations[score] * weight for score, weight in weights.items()
        ) * 100
        
        # Normalize to 0-100 range
        recommendations['final_score'] = np.clip(recommendations['final_score'], 0, 100)
        
        return recommendations

    def calculate_genre_match(self, movie_genres, preferred_genres):
        """Calculate genre matching score"""
        if not movie_genres or not preferred_genres:
            return 0.0
        
        movie_genre_list = [g.strip().lower() for g in movie_genres.split(',')]
        matches = sum(1 for genre in preferred_genres if any(genre in mg for mg in movie_genre_list))
        return matches / len(preferred_genres)

    def calculate_runtime_score(self, runtime, user_preferences):
        """Calculate runtime preference score"""
        if pd.isna(runtime):
            return 0.0
        
        min_runtime = user_preferences.get('min_runtime')
        max_runtime = user_preferences.get('max_runtime')
        
        # Convert string values to int/float if needed
        try:
            if min_runtime and isinstance(min_runtime, str):
                min_runtime = int(min_runtime) if min_runtime.strip() else None
            elif min_runtime and isinstance(min_runtime, str) and not min_runtime.strip():
                min_runtime = None
        except (ValueError, TypeError):
            min_runtime = None
            
        try:
            if max_runtime and isinstance(max_runtime, str):
                max_runtime = int(max_runtime) if max_runtime.strip() else None
            elif max_runtime and isinstance(max_runtime, str) and not max_runtime.strip():
                max_runtime = None
        except (ValueError, TypeError):
            max_runtime = None
        
        if min_runtime and max_runtime:
            if min_runtime <= runtime <= max_runtime:
                return 1.0
            else:
                return 0.0
        elif min_runtime:
            return 1.0 if runtime >= min_runtime else 0.0
        elif max_runtime:
            return 1.0 if runtime <= max_runtime else 0.0
        
        return 0.0

    def calculate_keyword_match(self, movie_keywords, preferred_keywords):
        """Calculate keyword matching score"""
        if not movie_keywords or not preferred_keywords:
            return 0.0
        
        movie_keyword_list = [k.strip().lower() for k in movie_keywords.split(',')]
        matches = sum(1 for keyword in preferred_keywords if any(keyword in mk for mk in movie_keyword_list))
        return matches / len(preferred_keywords)

    def calculate_company_match(self, movie_companies, preferred_companies):
        """Calculate production company matching score"""
        if not movie_companies or not preferred_companies:
            return 0.0
        
        movie_company_list = [c.strip().lower() for c in movie_companies.split(',')]
        matches = sum(1 for company in preferred_companies if any(company in mc for mc in movie_company_list))
        return matches / len(preferred_companies)

    def calculate_year_score(self, release_year, user_preferences):
        """Calculate year range matching score"""
        if pd.isna(release_year):
            return 0.0
        
        start_year = user_preferences.get('release_year_start')
        end_year = user_preferences.get('release_year_end')
        
        # Convert string values to int if needed
        try:
            if start_year and isinstance(start_year, str):
                start_year = int(start_year) if start_year.strip() else None
            elif start_year and isinstance(start_year, str) and not start_year.strip():
                start_year = None
        except (ValueError, TypeError):
            start_year = None
            
        try:
            if end_year and isinstance(end_year, str):
                end_year = int(end_year) if end_year.strip() else None
            elif end_year and isinstance(end_year, str) and not end_year.strip():
                end_year = None
        except (ValueError, TypeError):
            end_year = None
        
        if start_year and end_year:
            if start_year <= release_year <= end_year:
                return 1.0
            else:
                return 0.0
        elif start_year:
            return 1.0 if release_year >= start_year else 0.0
        elif end_year:
            return 1.0 if release_year <= end_year else 0.0
        
        return 0.0

    def fetch_tmdb_movie(self, title):
        """Fetch movie data from TMDb API"""
        try:
            search_url = f'{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={title}'
            search_response = requests.get(search_url)
            
            if search_response.status_code == 200:
                data = search_response.json()
                if data['results']:
                    movie = data['results'][0]
                    genres = ' '.join([g['name'] for g in movie.get('genre_ids', [])])
                    return {
                        'title': movie['title'],
                        'genres': genres,
                        'vote_average': movie['vote_average'],
                        'release_date': movie['release_date'],
                        'overview': movie['overview'] or '',
                        'original_language': movie['original_language']
                    }
            return None
        except Exception as e:
            logger.error(f"Error fetching TMDb movie: {e}")
            return None

    def log_search_history(self, username, user_preferences):
        """Log user search history"""
        try:
            conn = sqlite3.connect('history.db')
            cursor = conn.cursor()
            malaysia_time = datetime.now(malaysia_tz).strftime('%Y-%m-%d %H:%M:%S')
            
            cursor.execute('''
                INSERT INTO searches (
                    username, favorite_movie, genres, original_language, min_rating,
                    max_runtime, min_runtime, keywords, production_companies,
                    release_year_start, release_year_end, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                username, user_preferences.get('favorite_movie', ''),
                user_preferences.get('genres', ''), user_preferences.get('original_language', ''),
                user_preferences.get('min_rating', 0), user_preferences.get('max_runtime'),
                user_preferences.get('min_runtime'), user_preferences.get('keywords', ''),
                user_preferences.get('production_companies', ''), user_preferences.get('release_year_start'),
                user_preferences.get('release_year_end'), malaysia_time
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Logged search history for {username} at {malaysia_time}")
        except Exception as e:
            logger.error(f"Error logging search history: {e}")

    def get_available_options(self):
        """Get available filter options from the dataset"""
        try:
            # Get unique genres
            all_genres = set()
            for genres_str in self.df['genres_str'].dropna():
                genres = [g.strip() for g in genres_str.split(',') if g.strip()]
                all_genres.update(genres)
            
            # Get unique languages
            languages = sorted(self.df['original_language'].dropna().unique())
            
            # Get year range
            min_year = int(self.df['release_year'].min())
            max_year = int(self.df['release_year'].max())
            
            # Get runtime range
            min_runtime = int(self.df['runtime'].min())
            max_runtime = int(self.df['runtime'].max())
            
            # Get top production companies
            all_companies = set()
            for companies_str in self.df['production_companies_str'].dropna():
                companies = [c.strip() for c in companies_str.split(',') if c.strip()]
                all_companies.update(companies)
            
            return {
                'genres': sorted(list(all_genres)),
                'languages': languages,
                'year_range': {'min': min_year, 'max': max_year},
                'runtime_range': {'min': min_runtime, 'max': max_runtime},
                'top_companies': sorted(list(all_companies))[:100]  # Top 100 companies
            }
        except Exception as e:
            logger.error(f"Error getting available options: {e}")
            return {}

    def get_watchlist_movies(self, titles, username):
        """Get details for watchlist movies and sync with database"""
        try:
            conn = sqlite3.connect('history.db')
            cursor = conn.cursor()

            # Sync localStorage with database
            current_watchlist = set(titles)
            db_watchlist = set(row[0] for row in cursor.execute('SELECT movie_title FROM watchlist WHERE username = ?', (username,)).fetchall())
            
            # Add new items to database
            for title in current_watchlist - db_watchlist:
                movie = self.df[self.df['title'] == title].iloc[0] if title in self.df['title'].values else None
                movie_id = movie['id'] if movie is not None else None
                cursor.execute('INSERT INTO watchlist (username, movie_title, movie_id) VALUES (?, ?, ?)',
                            (username, title, movie_id))
            
            # Remove deleted items from database
            for title in db_watchlist - current_watchlist:
                cursor.execute('DELETE FROM watchlist WHERE username = ? AND movie_title = ?', (username, title))

            conn.commit()
            conn.close()

            df_filtered = self.df[self.df['title'].isin(titles)]
            
            result = {
                "movies": [{
                    'id': int(row['id']) if pd.notna(row['id']) else 0,
                    'title': row['title'],
                    'year': int(row['release_year']) if pd.notna(row['release_year']) else 'N/A',
                    'runtime': int(row['runtime']) if pd.notna(row['runtime']) else 'N/A',
                    'rating': float(row['vote_average']) if pd.notna(row['vote_average']) else 0.0,
                    'description': row['overview'] if pd.notna(row['overview']) else 'No description available',
                    'genres': row['genres_str'] if pd.notna(row['genres_str']) else 'N/A',
                    'language': row['original_language'] if pd.notna(row['original_language']) else 'N/A',
                    'keywords': row['keywords_str'] if pd.notna(row['keywords_str']) else 'N/A',
                    'production_companies': row['production_companies_str'] if pd.notna(row['production_companies_str']) else 'N/A',
                    'watch_link': f'https://www.themoviedb.org/movie/{int(row["id"])}' if pd.notna(row['id']) else 'https://www.themoviedb.org'
                } for _, row in df_filtered.iterrows()]
            }
            
            return result
        except Exception as e:
            logger.error(f"Error getting watchlist movies: {e}")
            return {"error": str(e)}

# Initialize the recommender
recommender = MovieRecommender()

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Redirect admin to admin dashboard
    if is_admin():
        return redirect(url_for('admin_dashboard'))
    
    # Get available options for filters
    options = recommender.get_available_options()
    options['language_map'] = LANGUAGE_MAP
    return render_template('index.html', options=options, language_map=LANGUAGE_MAP)

@app.route('/api/options')
def get_options():
    """API endpoint to get available filter options"""
    return jsonify(recommender.get_available_options())

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'username' not in session:
        return jsonify({'error': 'Please login first'}), 401
    
    try:
        data = request.get_json() if request.is_json else request.form
        
        user_preferences = {
            'favorite_movie': data.get('favorite_movie', ''),
            'genres': data.get('genres', ''),
            'original_language': data.get('original_language', ''),
            'min_rating': data.get('min_rating', 0),
            'max_runtime': data.get('max_runtime'),
            'min_runtime': data.get('min_runtime'),
            'keywords': data.get('keywords', ''),
            'production_companies': data.get('production_companies', ''),
            'release_year_start': data.get('release_year_start'),
            'release_year_end': data.get('release_year_end')
        }
        
        is_initial = request.headers.get('X-Initial-Load') == 'true'
        result = recommender.get_recommendations(user_preferences, is_initial)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Server error in recommend: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/watchlist', methods=['POST'])
def get_watchlist():
    if 'username' not in session:
        return jsonify({'error': 'Please login first'}), 401
    
    try:
        data = request.get_json()
        titles = data.get('titles', [])
        result = recommender.get_watchlist_movies(titles, session['username'])
        return jsonify(result)
    except Exception as e:
        logger.error(f"Server error in watchlist: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            return render_template('login.html', error="Username and password are required")
        
        # Check database authentication
        if recommender.authenticate_user(username, password):
            session['username'] = username
            
            # Redirect admin to admin dashboard
            if is_admin():
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid username or password")
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        if not username or not email or not password:
            return render_template('register.html', error="All fields are required")
        
        if len(username) < 3 or len(username) > 50:
            return render_template('register.html', error="Username must be 3-50 characters long")
        
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return render_template('register.html', error="Username can only contain letters, numbers, and underscores")
        
        if len(password) < 6:
            return render_template('register.html', error="Password must be at least 6 characters long")
        
        if password != confirm_password:
            return render_template('register.html', error="Passwords do not match")
        
        # Email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return render_template('register.html', error="Please enter a valid email address")
        
        # Create user
        result = recommender.create_user(username, email, password)
        
        if result['success']:
            return render_template('register.html', success="Account created successfully! You can now login.")
        else:
            return render_template('register.html', error=result['error'])
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    try:
        conn = sqlite3.connect('history.db')
        cursor = conn.cursor()
        
        # Check if searches table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='searches'")
        if not cursor.fetchone():
            logger.warning("Searches table not found, reinitializing database")
            recommender.initialize_database()
        
        cursor.execute('''
            SELECT favorite_movie, genres, original_language, min_rating, 
                   max_runtime, min_runtime, keywords, production_companies,
                   release_year_start, release_year_end, timestamp 
            FROM searches 
            WHERE username = ? 
            ORDER BY timestamp DESC
        ''', (session['username'],))
        
        history_data = cursor.fetchall()
        conn.close()
        logger.debug(f"History data fetched for {session['username']}: {history_data}")
        
        return render_template('history.html', history=history_data)
    except sqlite3.OperationalError as e:
        logger.error(f"Database error in history: {e}")
        return render_template('history.html', history=[], error=f"Database error: {str(e)}. Please ensure the database is initialized.")
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return render_template('history.html', history=[], error=f"Error loading history: {str(e)}")

@app.route('/watchlist')
def watchlist():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return render_template('watchlist.html')

@app.route('/admin')
def admin_dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if not is_admin():
        return redirect(url_for('index'))
    
    try:
        conn = sqlite3.connect('history.db')
        cursor = conn.cursor()
        
        # Get user statistics
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM searches')
        total_searches = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM watchlist')
        total_watchlist_items = cursor.fetchone()[0]
        
        # Get recent users
        cursor.execute('''
            SELECT username, email, created_at 
            FROM users 
            ORDER BY created_at DESC 
            LIMIT 10
        ''')
        recent_users = cursor.fetchall()
        
        # Get top searched movies
        cursor.execute('''
            SELECT favorite_movie, COUNT(*) as search_count
            FROM searches 
            WHERE favorite_movie != '' 
            GROUP BY favorite_movie 
            ORDER BY search_count DESC 
            LIMIT 10
        ''')
        top_movies = cursor.fetchall()
        
        # Get popular genres
        cursor.execute('''
            SELECT genres, COUNT(*) as genre_count
            FROM searches 
            WHERE genres != '' 
            GROUP BY genres 
            ORDER BY genre_count DESC 
            LIMIT 10
        ''')
        popular_genres = cursor.fetchall()
        
        conn.close()
        
        stats = {
            'total_users': total_users,
            'total_searches': total_searches,
            'total_watchlist_items': total_watchlist_items,
            'recent_users': recent_users,
            'top_movies': top_movies,
            'popular_genres': popular_genres
        }
        
        return render_template('admin.html', stats=stats)
        
    except Exception as e:
        logger.error(f"Error in admin dashboard: {e}")
        return render_template('admin.html', error=str(e))

@app.route('/admin_user')
def admin_user():
    if 'username' not in session or not is_admin():
        return redirect(url_for('login'))
    
    try:
        conn = sqlite3.connect('history.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT u.username, u.email, u.created_at,
                   COUNT(DISTINCT s.id) as search_count,
                   COUNT(DISTINCT w.id) as watchlist_count
            FROM users u
            LEFT JOIN searches s ON u.username = s.username
            LEFT JOIN watchlist w ON u.username = w.username
            GROUP BY u.username, u.email, u.created_at
            ORDER BY u.created_at DESC
        ''')
        
        users = cursor.fetchall()
        conn.close()
        
        return render_template('admin_user.html', users=users)
        
    except Exception as e:
        logger.error(f"Error in admin users: {e}")
        return render_template('admin_user.html', error=str(e))

@app.route('/admin/delete_user/<username>', methods=['POST'])
def admin_delete_user(username):
    if 'username' not in session or not is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    if username == 'admin':
        return jsonify({'error': 'Cannot delete admin user'}), 400
    
    try:
        conn = sqlite3.connect('history.db')
        cursor = conn.cursor()
        
        # Delete user's data
        cursor.execute('DELETE FROM watchlist WHERE username = ?', (username,))
        cursor.execute('DELETE FROM searches WHERE username = ?', (username,))
        cursor.execute('DELETE FROM users WHERE username = ?', (username,))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': f'User {username} deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
