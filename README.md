# 🎬 Movie Recommender CLI

A simple, command-line based movie recommender system built with Python and powered by user-based collaborative filtering. It uses cosine similarity to recommend movies to users based on the preferences of similar users.

---

## 📄 Project Structure

```
project-folder/
├── main.py                    # CLI entry point for the recommender system
├── recommender_model.py       # Core logic for data processing and recommendation
├── ratings.csv                # Ratings dataset (userId, movieId, rating, timestamp)
├── movies.csv                 # Movies dataset (movieId, title, genres)
├── README.md                  # Project documentation
└── __pycache__/               # Python bytecode cache
```

---

## ⚙️ How It Works

### 1. Data Preparation (`load_and_prepare_data()`)
- Loads `ratings.csv` and `movies.csv`
- Merges them on `movieId`
- Drops unnecessary columns like `timestamp`
- Returns both the merged dataset and the original `movies` dataframe

> **Why return `movies` separately?**
> Although the merged dataset already includes movie information, we return the original `movies` dataframe separately for modularity and flexibility. It may be used independently for mapping, filtering, or reference elsewhere in the code without the overhead of ratings.

### 2. Create User-Movie Matrix (`create_user_movie_matrix()`)
- Transforms data into a matrix where rows = users and columns = movies
- Cells contain the rating each user gave to a movie
- Missing ratings are filled with 0

### 3. Compute Similarity (`compute_user_similarity()`)
- Calculates cosine similarity between users based on their ratings
- Returns a similarity matrix where each cell (i,j) is the similarity between user i and user j

### 4. Generate Recommendations (`recommender_logic()`)
- For a given user, finds the top N similar users
- Aggregates their ratings using weighted average (based on similarity)
- Recommends movies the target user hasn’t rated yet
- Returns top N predicted movies with titles and predicted ratings

---

## 🔧 How to Run

Make sure you have Python 3.x installed. Then, run:

```bash
python main.py
```

You will be prompted to enter your User ID and the number of recommendations you want.

---

## 📅 Example Output

```text
🎥 Welcome to the Movie Recommender CLI!

Enter your User ID (or -1 to exit): 3
How many movie recommendations would you like? 5

🎥 Top Recommendations for You:
- The Matrix (Predicted Rating: 4.65)
- The Godfather (Predicted Rating: 4.52)
- Inception (Predicted Rating: 4.47)
...
```

---

## ⚡ Features

- User-based collaborative filtering
- Cosine similarity computation
- Interactive CLI with error handling
- Modular and reusable codebase

---

## 🚀 Future Improvements

- Add content-based filtering (using genres)
- Hybrid recommender system
- Web interface with Flask or Streamlit
- Store user feedback to improve recommendations

---

## ✅ Requirements

- Python 3.x
- `pandas`
- `numpy`
- `scikit-learn`

Install dependencies via:
```bash
pip install pandas numpy scikit-learn
```

---

## 👥 Credits

Created by [Hitansh Bhagtani] as part of a CLI-based recommender system project using the MovieLens dataset.

---



