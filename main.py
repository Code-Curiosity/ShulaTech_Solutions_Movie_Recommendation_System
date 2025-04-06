from recommender_model import (
    load_and_prepare_data,
    create_user_movie_matrix,
    compute_user_similarity,
    recommender_logic
)

def main():
    print("🎬 Welcome to the Movie Recommender CLI!\n")

    # Load and prepare data
    data, movies = load_and_prepare_data()
    user_movie_matrix = create_user_movie_matrix(data)

    # Get user similarity matrix from recommender_model.py
    similarity_df = compute_user_similarity(user_movie_matrix)

    while True:
        try:
            user_id = int(input("Enter your User ID (or -1 to exit): "))
            if user_id == -1:
                print("👋 Exiting the recommender. Goodbye!")
                break

            if user_id not in user_movie_matrix.index:
                print("⚠️ User ID not found. Try again.")
                continue

            n = int(input("How many movie recommendations would you like? "))
        except ValueError:
            print("❌ Please enter valid numeric inputs.")
            continue

        recommendations = recommender_logic(user_id, user_movie_matrix, similarity_df, movies, n)
        
        if recommendations.empty:
            print("😕 No recommendations found.")
        else:
            print("\n🎥 Top Recommendations for You:\n")
            for idx, row in recommendations.iterrows():
                print(f"- {row['title']} (Predicted Rating: {row['Predicted Rating']:.2f})")
            print()

if __name__ == "__main__":
    main()