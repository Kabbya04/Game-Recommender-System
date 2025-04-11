import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Omni")


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

st.title("Omni : Your Own Game Recommender System")

selected_game_name = st.selectbox(
"Enter game name that you liked:",
games['Title'].values
)

recommendation = recommend(selected_game_name,cv_similarity)[0]
cv_score = recommend(selected_game_name,cv_similarity)[1]
tf_idf_score = recommend(selected_game_name,tf_idf_similarity)[1]
cv_prediction = recommend(selected_game_name,cv_similarity)[2]
tf_idf_prediction = recommend(selected_game_name,tf_idf_similarity)[2]

if st.button("Bag of Words Recommendation"):

    count = 0

    for i in recommendation:
        label = "Highly recommended" if cv_prediction[count] == 1 else "Casually recommended"
        idx=count+1
        st.markdown(f"{idx}.**Game:** {i}  \n**CV Similarity Score:** {cv_score[count]}  \n**Status:** {label}")
        count+=1

if st.button("TF_IDF Recommendation"):

    count = 0

    for i in recommendation:
        label = "Highly recommended" if tf_idf_prediction[count] == 1 else "Casually recommended"
        idx=count+1
        st.markdown(f"{idx}.**Game:** {i}  \n**TF_IDF Similarity Score:** {tf_idf_score[count]}  \n**Status:** {label}")
        count+=1