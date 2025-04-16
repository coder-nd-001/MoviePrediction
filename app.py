import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open(r"C:\Users\Nagesh\Desktop\yash\projectmovie\model.pkl", 'rb') as file:
    model = pickle.load(file)

# Streamlit App Title
st.title("🎬 Movie Success Prediction App")
st.write("Enter movie details below to predict the estimated **gross revenue** 💰")

# Input Fields
director_name = st.text_input("🎬 Director Name")
actor_1_name = st.text_input("🎭 Main Actor Name")
genres = st.text_input("🎞️ Genres (e.g., Action|Adventure)")
duration = st.number_input("⏱️ Duration (in minutes)", min_value=0)
budget = st.number_input("💰 Budget (in USD)", min_value=0)
title_year = st.number_input("📅 Year of Release", min_value=1900, max_value=2025)

# Predict Button
if st.button("Predict"):
    if not (director_name and actor_1_name and genres and duration > 0 and budget > 0 and title_year):
        st.error("Please fill out all fields correctly before predicting.")
    else:
        # Prepare input as DataFrame
        input_df = pd.DataFrame({
            'director_name': [director_name],
            'duration': [duration],
            'actor_1_name': [actor_1_name],
            'budget': [budget],
            'genres': [genres],
            'title_year': [title_year]
        })

        try:
            # Make prediction
            prediction = model.predict(input_df)
            st.success(f"🎉 Estimated Gross Revenue: **${prediction[0]:,.2f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
