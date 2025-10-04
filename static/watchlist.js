document.addEventListener('DOMContentLoaded', function() {
    console.log('Watchlist page loaded'); // Debug log
    const watchlistContent = document.getElementById('watchlist-content');

    function loadWatchlist() {
        console.log('Loading watchlist...'); // Debug log
        
        // Get watchlist from localStorage
        let watchlist;
        try {
            watchlist = JSON.parse(localStorage.getItem('watchlist') || '[]');
            console.log('Watchlist from localStorage:', watchlist); // Debug log
        } catch (error) {
            console.error('Error parsing watchlist from localStorage:', error);
            watchlist = [];
        }
        
        if (!watchlist || watchlist.length === 0) {
            watchlistContent.innerHTML = `
                <div class="bg-gray-800 p-4 rounded-lg text-center">
                    <p>Your watchlist is empty. Add movies from the recommendations page!</p>
                </div>
            `;
            return;
        }

        // Check if we have the API endpoint available
        fetch('/api/watchlist', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ titles: watchlist })
        })
        .then(response => {
            console.log('API response status:', response.status); // Debug log
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('API response data:', data); // Debug log
            
            if (data.error) {
                watchlistContent.innerHTML = `
                    <div class="bg-red-600 p-4 rounded-lg text-center">
                        <p>Error loading watchlist: ${data.error}</p>
                    </div>
                `;
                return;
            }

            displayWatchlistMovies(data.movies);
        })
        .catch(error => {
            console.error('Error loading watchlist:', error);
            
            // Fallback: display basic watchlist without movie details
            watchlistContent.innerHTML = `
                <div class="bg-yellow-600 p-4 rounded-lg text-center mb-4">
                    <p>Unable to load full movie details. Showing basic watchlist:</p>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    ${watchlist.map(title => `
                        <div class="bg-gray-800 rounded-lg p-4">
                            <div class="flex justify-between items-center mb-2">
                                <h4 class="font-bold text-lg">${title}</h4>
                                <button onclick="removeFromWatchlist('${title.replace(/'/g, "\\'")}')" 
                                    class="text-xs bg-red-600 hover:bg-red-700 px-2 py-1 rounded">
                                    Remove
                                </button>
                            </div>
                            <p class="text-sm text-gray-400">Movie details unavailable</p>
                        </div>
                    `).join('')}
                </div>
            `;
        });
    }

    function displayWatchlistMovies(movies) {
        let html = `
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        `;

        movies.forEach(movie => {
            const rating = movie.rating ? parseFloat(movie.rating).toFixed(1) : 'N/A';
            const year = movie.year || 'N/A';
            const runtime = movie.runtime && movie.runtime !== 'N/A' ? `${movie.runtime} min` : 'N/A';
            const genres = movie.genres || 'N/A';
            const description = movie.description || 'No description available';

            html += `
                <div class="bg-gray-800 rounded-lg p-4">
                    <div class="flex justify-between items-center mb-2">
                        <h4 class="font-bold text-lg">${movie.title}</h4>
                        <button onclick="removeFromWatchlist('${movie.title.replace(/'/g, "\\'")}')" 
                            class="text-xs bg-red-600 hover:bg-red-700 px-2 py-1 rounded">
                            Remove
                        </button>
                    </div>
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-sm">${year} • ${runtime}</span>
                    </div>
                    <div class="flex items-center mb-2">
                        <span class="text-yellow-300 mr-1">⭐</span>
                        <span class="text-sm">${rating}</span>
                    </div>
                    <p class="text-sm text-gray-200 mb-2">${genres}</p>
                    <p class="text-sm text-gray-100 mb-3 line-clamp-3">${description}</p>
                    <a href="${movie.watch_link}" target="_blank" class="text-blue-400 hover:underline text-sm">
                        🎬 More Info
                    </a>
                </div>
            `;
        });

        html += '</div>';
        watchlistContent.innerHTML = html;
    }

    window.removeFromWatchlist = function(movieTitle) {
        console.log('Removing from watchlist:', movieTitle); // Debug log
        
        let watchlist = JSON.parse(localStorage.getItem('watchlist') || '[]');
        watchlist = watchlist.filter(title => title !== movieTitle);
        localStorage.setItem('watchlist', JSON.stringify(watchlist));
        
        // Also update in-memory data if it exists
        if (window.watchlistData) {
            window.watchlistData = window.watchlistData.filter(title => title !== movieTitle);
        }
        
        loadWatchlist();
    };

    // Load watchlist on page load
    loadWatchlist();
});