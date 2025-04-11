import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Omni",page_icon="./logo/monogram-hq.png")

# Add custom CSS for the background (from your previous setup)
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom, rgba(94, 89, 137, 0.9), rgba(180, 166, 171, 0.9)) !important;
        color: white !important;
        margin: 0;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom, rgba(94, 89, 137, 0.9), rgba(180, 166, 171, 0.9)) !important;
        color: white !important;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
    }
    [data-testid="stAppViewContainer"] > .main {
        background: linear-gradient(to bottom, rgba(94, 89, 137, 0.9), rgba(180, 166, 171, 0.9)) !important;
        color: white !important;
        flex: 1;
    }

    /* Centered image styling */
    .centered-image {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }
    .centered-image img {
        max-width: 100%;
        height: auto;
        width: 300px;
    }

    /* Selectbox container styling */
    .selectbox-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
        position: relative;  /* For positioning the custom arrow */
    }
    /* Selectbox styling to match buttons */
    [data-testid="stSelectbox"] {
        width: 300px;  /* Ensure the container matches the selectbox width */
    }
    [data-testid="stSelectbox"] select {
        background-color: black !important;
        color: white !important;
        border: 2px solid black !important;
        padding: 10px 30px 10px 20px;  /* Extra padding on the right for the arrow */
        border-radius: 5px;
        transition: all 0.3s ease;
        width: 100%;  /* Use 100% width to fill the container */
        appearance: none;
        cursor: pointer;
        box-shadow: none !important;  /* Remove any default shadow */
    }
    [data-testid="stSelectbox"] select:hover {
        color: white !important;
        border-color: white !important;
        background-color: black !important;
    }
    [data-testid="stSelectbox"] select:active {
        background-color: white !important;
        border-color: white !important;
        color: black !important;
    }
    [data-testid="stSelectbox"] select:focus {
        background-color: black !important;
        color: white !important;
        border: 2px solid black !important;  /* Explicitly set border */
        border-color: black !important;  /* Fallback for border-color */
        outline: none !important;  /* Remove default focus outline */
        box-shadow: none !important;  /* Remove any focus shadow */
    }
    /* Style the dropdown options */
    [data-testid="stSelectbox"] select option {
        background-color: black !important;
        color: white !important;
    }

    /* Button styling */
    .stButton > button {
        /* Base form: white text on black background */
        background-color: black;
        color: white;
        border: 2px solid black;  /* Border matches the background initially */
        padding: 10px 20px;
        border-radius: 5px;
        margin: 10px 0;
        transition: all 0.3s ease;  /* Smooth transition for all changes */
    }
    .stButton > button:hover {
        /* Hover: text and border turn white */
        color: white;
        border-color: white;
        background-color: black;  /* Background stays black */
    }
    .stButton > button:active {
        /* Click (active): background and border turn white, text turns black */
        background-color: white;
        border-color: white;
        color: black;
    }
    .stButton > button:focus {
        /* After clicking: explicitly revert to base form */
        background-color: black !important;
        color: white !important;
        border-color: black !important;
        outline: none !important;  /* Remove default focus outline */
    }
    </style>
    """,
    unsafe_allow_html=True
)


def recommend(game, similarity, svm_model_path='svm_model.pkl', n_recommendations=5):
    # Load the trained SVM model
    sgd_svm = joblib.load(svm_model_path)

    # Find the index of the input game
    game_index = games[games['Title'] == game].index[0]
    
    # Get similarity scores
    distances = similarity[game_index]
    
    # Get top n recommendations (excluding the input game itself)
    game_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:n_recommendations+1]
    
    # Extract titles and similarity scores
    recommended_games = []
    similarity_scores = []
    svm_predictions = []
    
    # SVM dataset game_likelihood (e.g., game_likelihood) has the features needed for prediction
    svm_features = ['Rating', 'Times Listed', 'Plays', 'Playing', 'Backlogs', 'Wishlist']
    
    for i in game_list:
        title = games.iloc[i[0]]['Title']
        score = i[1]
        recommended_games.append(title)
        similarity_scores.append(score)
        
        # Get SVM prediction by mapping title to the SVM dataset game_likelihood
        game_row = likelihood[likelihood['Title'] == title]
        if not game_row.empty:
            features = game_row[svm_features]
            prediction = sgd_svm.predict(features)[0]  # Predict for this game
            svm_predictions.append(prediction)
        else:
            svm_predictions.append(None)  # Handle missing titles
    
    # Return recommendations and scores
    return recommended_games, similarity_scores, svm_predictions

games_dict = pickle.load(open('games_dict.pkl', 'rb'))
games = pd.DataFrame(games_dict)

likelihood_dict = pickle.load(open('likelihood_dict.pkl', 'rb'))
likelihood = pd.DataFrame(likelihood_dict)

cv_similarity = pickle.load(open('cv_similarity.pkl', 'rb'))
tf_idf_similarity = pickle.load(open('tf_idf_similarity.pkl', 'rb'))


# Add the centered image as the title
st.markdown('<div class="centered-image">', unsafe_allow_html=True)
st.image("logo/omni-high-resolution-logo-transparent.png", use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)

#game select box
selected_game_name = st.selectbox(
"Enter game name that you liked:",
games['Title'].values
)

cv_recommendation = recommend(selected_game_name,cv_similarity)[0]
tf_idf_recommendation = recommend(selected_game_name,tf_idf_similarity)[0]
cv_score = recommend(selected_game_name,cv_similarity)[1]
tf_idf_score = recommend(selected_game_name,tf_idf_similarity)[1]
cv_prediction = recommend(selected_game_name,cv_similarity)[2]
tf_idf_prediction = recommend(selected_game_name,tf_idf_similarity)[2]

with st.container():
    if st.button("Bag of Words Recommendation"):

        count = 0

        for i in cv_recommendation:
            label = "Highly recommended" if cv_prediction[count] == 1 else "Casually recommended"
            idx=count+1
            st.markdown(f"{idx}.**Game:** {i}  \n**CV Similarity Score:** {cv_score[count]}  \n**Status:** {label}")
            count+=1

    if st.button("TF_IDF Recommendation"):

        count = 0

        for i in tf_idf_recommendation:
            label = "Highly recommended" if tf_idf_prediction[count] == 1 else "Casually recommended"
            idx=count+1
            st.markdown(f"{idx}.**Game:** {i}  \n**TF_IDF Similarity Score:** {tf_idf_score[count]}  \n**Status:** {label}")
            count+=1

# Footer message
st.markdown(
    '<p style="text-align: center; font-size: 16px; color: #000000; margin-top: 20px;">Welcome to Omni! Enter a game name to get personalized recommendations.</p>',
    unsafe_allow_html=True
)