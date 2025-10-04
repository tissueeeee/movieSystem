document.addEventListener('DOMContentLoaded', function() {
    const recommendationSection = document.getElementById('recommendation-section');
    const recommendationContent = document.getElementById('recommendation-content');
    const loading = document.querySelector('.loading');
    const form = document.getElementById('recommendForm');
    const genrePills = document.getElementById('genre-pills');
    const selectedGenresInput = document.getElementById('selected-genres');
    const favoriteMovieInput = document.getElementById('favorite_movie');

    // Initialize genre pill selection (multiple selection)
    genrePills.addEventListener('click', function(e) {
        if (e.target.classList.contains('genre-pill')) {
            // Handle "Any Genre" selection
            if (e.target.getAttribute('data-genre') === '') {
                // Clear all other selections
                genrePills.querySelectorAll('.genre-pill').forEach(pill => {
                    pill.classList.remove('selected');
                });
                e.target.classList.add('selected');
                selectedGenresInput.value = '';
            } else {
                // Remove "Any Genre" selection if another genre is selected
                const anyGenrePill = genrePills.querySelector('.genre-pill[data-genre=""]');
                if (anyGenrePill) {
                    anyGenrePill.classList.remove('selected');
                }
                
                // Toggle current selection
                e.target.classList.toggle('selected');
                
                // Update hidden input
                const selectedGenres = Array.from(genrePills.querySelectorAll('.genre-pill.selected'))
                    .map(pill => pill.getAttribute('data-genre'))
                    .filter(genre => genre);
                selectedGenresInput.value = selectedGenres.join(',');
            }
        }
    });

    // Movie title autocomplete functionality
    let movieTitles = [];
    
    // Extract movie titles from the options or create a simple list
    if (typeof options !== 'undefined' && options.movies) {
        movieTitles = options.movies.map(movie => movie.title);
    }

    // Simple autocomplete for favorite movie input
    favoriteMovieInput.addEventListener('input', function(e) {
        const value = e.target.value.toLowerCase();
        
        // Remove existing suggestions
        const existingSuggestions = document.getElementById('movie-suggestions');
        if (existingSuggestions) {
            existingSuggestions.remove();
        }
        
        if (value.length > 2) {
            // Create suggestions dropdown
            const suggestions = document.createElement('div');
            suggestions.id = 'movie-suggestions';
            suggestions.style.cssText = `
                position: absolute;
                top: 100%;
                left: 0;
                right: 0;
                background: white;
                border: 1px solid #ccc;
                border-top: none;
                max-height: 200px;
                overflow-y: auto;
                z-index: 1000;
            `;
            
            // Filter and display suggestions
            const filteredTitles = movieTitles
                .filter(title => title.toLowerCase().includes(value))
                .slice(0, 10);
            
            filteredTitles.forEach(title => {
                const suggestion = document.createElement('div');
                suggestion.textContent = title;
                suggestion.style.cssText = `
                    padding: 8px;
                    cursor: pointer;
                    border-bottom: 1px solid #eee;
                `;
                suggestion.addEventListener('mouseenter', () => {
                    suggestion.style.backgroundColor = '#f0f0f0';
                });
                suggestion.addEventListener('mouseleave', () => {
                    suggestion.style.backgroundColor = 'white';
                });
                suggestion.addEventListener('click', () => {
                    favoriteMovieInput.value = title;
                    suggestions.remove();
                });
                suggestions.appendChild(suggestion);
            });
            
            // Position suggestions relative to input
            const inputContainer = favoriteMovieInput.parentElement;
            inputContainer.style.position = 'relative';
            inputContainer.appendChild(suggestions);
        }
    });

    // Close suggestions when clicking outside
    document.addEventListener('click', function(e) {
        if (!e.target.closest('#favorite_movie')) {
            const suggestions = document.getElementById('movie-suggestions');
            if (suggestions) {
                suggestions.remove();
            }
        }
    });

    // Load initial random recommendations without logging history
    loadInitialRecommendations();

    // Handle form submission
    form.addEventListener('submit', handleFormSubmission);

    function showLoading() {
        loading.style.display = 'block';
        recommendationSection.classList.add('show');
        recommendationContent.innerHTML = '';
    }

    function hideLoading() {
        loading.style.display = 'none';
    }

    function loadInitialRecommendations() {
        showLoading();
        
        // Create initial random preferences
        const formData = new FormData(form);
        
        fetch('/recommend', {
            method: 'POST',
            body: new URLSearchParams(formData),
            headers: { 'X-Initial-Load': 'true' }
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            displayRecommendations(data);
        })
        .catch(error => {
            hideLoading();
            console.error('Error loading initial recommendations:', error);
            recommendationContent.innerHTML = `
                <div class="text-center p-4">
                    <p class="error">Unable to load recommendations. Please try again.</p>
                </div>
            `;
        });
    }

    function handleFormSubmission(e) {
        e.preventDefault();
        showLoading();
        
        const formData = new FormData(form);
        const processedData = new FormData();

        // Process form data to ensure numeric values
        for (let [key, value] of formData.entries()) {
            // Skip empty values
            if (value === '') {
                continue;
            }

            // Convert runtime values to numbers
            if (key === 'min_runtime' || key === 'max_runtime') {
                const numValue = parseInt(value);
                if (!isNaN(numValue)) {
                    processedData.append(key, numValue);
                }
            }
            // Convert year values to numbers
            else if (key === 'release_year_start' || key === 'release_year_end') {
                const numValue = parseInt(value);
                if (!isNaN(numValue)) {
                    processedData.append(key, numValue);
                }
            }
            // Handle min_rating as float
            else if (key === 'min_rating') {
                const numValue = parseFloat(value);
                if (!isNaN(numValue)) {
                    processedData.append(key, numValue);
                }
            }
            // Keep other values as is
            else {
                processedData.append(key, value);
            }
        }
        
        fetch('/recommend', {
            method: 'POST',
            body: new URLSearchParams(processedData),
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            displayRecommendations(data);
            recommendationSection.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            hideLoading();
            console.error('Error getting recommendations:', error);
            recommendationContent.innerHTML = `
                <div class="text-center p-4">
                    <p class="error">Error getting recommendations. Please try again.</p>
                </div>
            `;
        });
    }

    function displayRecommendations(data) {
    recommendationSection.classList.add('show');
    
    if (data.error) {
        recommendationContent.innerHTML = `
            <div class="text-center p-4">
                <p class="error">${data.error}</p>
            </div>
        `;
        return;
    }

    if (!data.recommendations || data.recommendations.length === 0) {
        recommendationContent.innerHTML = `
            <div class="text-center p-4">
                <p class="error">No recommendations found. Please try adjusting your preferences.</p>
            </div>
        `;
        return;
    }

    let html = `
        <div class="mb-6 text-center">
            <p class="text-gray-300 text-lg">Found ${data.recommendations.length} movies 
            ${data.selected_movie ? `similar to "${data.selected_movie}"` : 'matching your preferences'}
            ${data.total_filtered ? `from ${data.total_filtered} filtered results` : ''}</p>
        </div>
        <div class="recommendation-grid">
    `;

    data.recommendations.forEach(movie => {
        const matchPercentage = Math.round(movie.match_percentage || 0);
        const rating = movie.rating ? parseFloat(movie.rating).toFixed(1) : 'N/A';
        const year = movie.year || 'N/A';
        const runtime = movie.runtime && movie.runtime !== 'N/A' ? `${movie.runtime} min` : 'N/A';
        const genres = movie.genres || 'N/A';
        const description = movie.description || 'No description available';
        
        html += `
            <div class="movie-container">
                <div class="movie-header">
                    <h3 class="movie-title">${movie.title}</h3>
                    <span class="match-badge">${matchPercentage}% Match</span>
                </div>
                
                <div class="movie-meta">
                    <div class="movie-year-runtime">
                        <span>${year}</span>
                        <span>•</span>
                        <span>${runtime}</span>
                    </div>
                    <div class="movie-rating">
                        <span>⭐</span>
                        <span>${rating}</span>
                    </div>
                </div>
                
                <div class="movie-genres">${genres}</div>
                
                <p class="movie-description">${description}</p>
                
                <div class="movie-actions">
                    <a href="${movie.watch_link}" target="_blank" class="more-info-link">
                        🎬 More Info
                    </a>
                    <button onclick="addToWatchlist('${movie.title.replace(/'/g, "\\'")}')" class="watchlist-btn">
                        + Watchlist
                    </button>
                </div>
            </div>
        `;
    });

    html += '</div>';
    recommendationContent.innerHTML = html;
}
 
    // Additional utility functions
    
    window.addToWatchlist = function(movieTitle) {
        // Use in-memory storage instead of localStorage
        if (!window.watchlistData) {
            window.watchlistData = [];
        }
        
        if (!window.watchlistData.includes(movieTitle)) {
            window.watchlistData.push(movieTitle);
            
            // Also sync with localStorage for watchlist page
            const existingWatchlist = JSON.parse(localStorage.getItem('watchlist') || '[]');
            if (!existingWatchlist.includes(movieTitle)) {
                existingWatchlist.push(movieTitle);
                localStorage.setItem('watchlist', JSON.stringify(existingWatchlist));
            }
            
            // Show feedback with improved styling
            const button = event.target;
            const originalText = button.textContent;
            const originalClass = button.className;
            
            button.textContent = '✓ Added';
            button.className = originalClass + ' added';
            
            setTimeout(() => {
                button.textContent = originalText;
                button.className = originalClass;
            }, 2000);
        } else {
            // Movie already in watchlist
            const button = event.target;
            const originalText = button.textContent;
            
            button.textContent = '✓ Already Added';
            
            setTimeout(() => {
                button.textContent = originalText;
            }, 2000);
        }
    };

    // Initialize watchlist data from localStorage on page load
    function initializeWatchlist() {
        const storedWatchlist = JSON.parse(localStorage.getItem('watchlist') || '[]');
        window.watchlistData = storedWatchlist;
    }
    
    // Call this function after your existing initialization
    initializeWatchlist();

    // Form validation
    function validateForm() {
        const minRuntime = parseInt(document.querySelector('input[name="min_runtime"]').value) || 0;
        const maxRuntime = parseInt(document.querySelector('input[name="max_runtime"]').value) || 0;
        const startYear = parseInt(document.querySelector('select[name="release_year_start"]').value) || 0;
        const endYear = parseInt(document.querySelector('select[name="release_year_end"]').value) || 0;
        
        if (minRuntime && maxRuntime && minRuntime > maxRuntime) {
            alert('Minimum runtime cannot be greater than maximum runtime');
            return false;
        }
        
        if (startYear && endYear && startYear > endYear) {
            alert('Start year cannot be later than end year');
            return false;
        }
        
        return true;
    }

    // Add validation to form submission
    form.addEventListener('submit', function(e) {
        if (!validateForm()) {
            e.preventDefault();
            return false;
        }
    });

    // Initialize "Any Genre" as selected by default
    const anyGenrePill = genrePills.querySelector('.genre-pill[data-genre=""]');
    if (anyGenrePill) {
        anyGenrePill.classList.add('selected');
    }

    // Handle runtime preference conversion
    document.querySelector('select[name="runtime_preference"]').addEventListener('change', function(e) {
        const minRuntimeInput = document.querySelector('input[name="min_runtime"]');
        const maxRuntimeInput = document.querySelector('input[name="max_runtime"]');
        
        switch(e.target.value) {
            case 'short':
                minRuntimeInput.value = '0';
                maxRuntimeInput.value = '90';
                break;
            case 'medium':
                minRuntimeInput.value = '90';
                maxRuntimeInput.value = '150';
                break;
            case 'long':
                minRuntimeInput.value = '150';
                maxRuntimeInput.value = '999';
                break;
            default:
                minRuntimeInput.value = '';
                maxRuntimeInput.value = '';
        }
    });

    // Handle release period conversion
    document.querySelector('select[name="release_period"]').addEventListener('change', function(e) {
        const startYearSelect = document.querySelector('select[name="release_year_start"]');
        const endYearSelect = document.querySelector('select[name="release_year_end"]');
        
        switch(e.target.value) {
            case '2020s':
                startYearSelect.value = '2020';
                endYearSelect.value = '2025';
                break;
            case '2010s':
                startYearSelect.value = '2010';
                endYearSelect.value = '2019';
                break;
            case '2000s':
                startYearSelect.value = '2000';
                endYearSelect.value = '2009';
                break;
            case '1990s':
                startYearSelect.value = '1990';
                endYearSelect.value = '1999';
                break;
            case '1980s':
                startYearSelect.value = '1980';
                endYearSelect.value = '1989';
                break;
            case '1970s':
                startYearSelect.value = '1970';
                endYearSelect.value = '1979';
                break;
            case 'classic':
                startYearSelect.value = '1900';
                endYearSelect.value = '1969';
                break;
            default:
                startYearSelect.value = '';
                endYearSelect.value = '';
        }
    });
});

    //  popular movies functionality
    document.addEventListener('DOMContentLoaded', function() {
        // Load More Button Functionality
        const loadMoreBtn = document.querySelector('.load-more-btn');
        if (loadMoreBtn) {
            loadMoreBtn.addEventListener('click', function() {
                // Add loading state
                this.innerHTML = '<div class="inline-block animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>Loading...';
                
                // Simulate API call
                setTimeout(() => {
                    // Reset button text
                    this.innerHTML = 'Load More Movies';
                    
                    // Here you would typically load more movies from your backend
                    console.log('Loading more popular movies...');
                }, 1500);
            });
        }

        // Add to Watchlist Functionality
        const quickAddBtns = document.querySelectorAll('.quick-add-btn');
        quickAddBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                const originalText = this.innerHTML;
                this.innerHTML = '✓ Added';
                this.style.background = 'var(--neon-green)';
                
                setTimeout(() => {
                    this.innerHTML = originalText;
                    this.style.background = 'var(--accent-gradient)';
                }, 2000);
            });
        });
    });