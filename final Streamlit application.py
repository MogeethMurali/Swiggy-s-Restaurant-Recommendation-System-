import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

# --- Load Assets ---
try:
    image = Image.open(r"C:\Users\Mogeeth.M\Downloads\swiggy-logo-png_seeklogo-348257.png")
    st.image(image, use_column_width=True)
except FileNotFoundError:
    st.warning("Swiggy logo not found. Skipping logo.")

# --- Load Data & Encoders ---
try:
    cleaned_df = pd.read_csv(r"C:\Users\Mogeeth.M\Downloads\swiggy\env\cleaned_data.csv")
    encoded_df = pd.read_csv(r"C:\Users\Mogeeth.M\Downloads\swiggy\env\encoded_data.csv")
    with open(r"C:\Users\Mogeeth.M\Downloads\swiggy\env\encoder.pkl", "rb") as f:
        ohe, mlb = pickle.load(f)
except Exception as e:
    st.error(f"❌ Failed to load data or encoders: {e}")
    st.stop()

# --- Streamlit Tabs ---
tab1, tab2 = st.tabs(["🔍 Recommendations", "📊 EDA Insights"])

# === TAB 1: Recommendations ===
with tab1:
    st.title("🍽 Swiggy Restaurant Recommender")
    st.markdown("Get personalized restaurant recommendations based on your preferences!")

    city_input = st.selectbox("📍 Select City", cleaned_df['city'].unique())
    cuisine_input = st.multiselect("🍱 Select Cuisine(s)", mlb.classes_)
    rating_input = st.slider("⭐ Minimum Rating", 0.0, 5.0, 3.5, 0.1)
    cost_input = st.slider("💰 Maximum Cost", 100, 1000, 500, 50)

    if st.button("🔍 Get Recommendations"):

        if not cuisine_input:
            st.warning("Please select at least one cuisine.")
            st.stop()

        # Encode input
        try:
            city_vec = ohe.transform([[city_input]])
            cuisine_vec = mlb.transform([cuisine_input])
        except Exception as e:
            st.error(f"Encoding Error: {e}")
            st.stop()

        # Combine user features
        user_features = np.hstack((city_vec, cuisine_vec, [[rating_input, 100, cost_input]]))  # Dummy vote count

        # Filter by city
        city_mask = cleaned_df['city'] == city_input
        filtered_encoded = encoded_df[city_mask]
        filtered_cleaned = cleaned_df[city_mask].reset_index(drop=True)

        if filtered_encoded.empty:
            st.info("No data found for selected city.")
            st.stop()

        # Normalize
        scaler = StandardScaler()
        filtered_scaled = scaler.fit_transform(filtered_encoded)
        user_scaled = scaler.transform(user_features)

        # Similarity
        similarities = cosine_similarity(filtered_scaled, user_scaled)
        top_indices = np.argsort(similarities.flatten())[::-1][:5]

        st.subheader("🎯 Top Recommended Restaurants:")
        for i, idx in enumerate(top_indices):
            row = filtered_cleaned.iloc[idx]
            st.markdown(f"### {i+1}. 🍴 {row['name']}")
            st.write(f"⭐ **Rating**: {row['rating']} | 💬 **Votes**: {row['rating_count']} | 💰 **Cost**: ₹{row['cost']}")
            st.write(f"📍 **Address**: {row['address']}")
            if isinstance(row['link'], str) and row['link'].startswith("http"):
                st.markdown(f"[🔗 View on Swiggy]({row['link']})")
            st.markdown("---")

# === TAB 2: EDA ===
with tab2:
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("📋 Dataset Overview")
    st.write("Number of restaurants:", cleaned_df.shape[0])
    st.write("Number of unique cities:", cleaned_df['city'].nunique())

    # Cuisine count
    all_cuisines = [c for sublist in cleaned_df['cuisine'].apply(eval) for c in sublist]
    st.write("Number of unique cuisines:", len(set(all_cuisines)))

    st.subheader("📌 Missing Value Summary")
    st.dataframe(cleaned_df.isnull().sum())

    st.subheader("🏙️ Top Cities by Restaurant Count")
    top_cities = cleaned_df['city'].value_counts().head(10)
    st.bar_chart(top_cities)

    st.subheader("🍱 Most Popular Cuisines")
    top_cuisines = Counter(all_cuisines).most_common(10)
    top_cuisine_df = pd.DataFrame(top_cuisines, columns=['Cuisine', 'Count'])
    st.bar_chart(top_cuisine_df.set_index('Cuisine'))

    st.subheader("💰 Cost Distribution")
    fig1, ax1 = plt.subplots()
    cleaned_df['cost'].hist(bins=30, ax=ax1, color='orange', edgecolor='black')
    ax1.set_xlabel("Cost for 2")
    ax1.set_ylabel("Number of Restaurants")
    st.pyplot(fig1)

    st.subheader("⭐ Rating Distribution")
    fig2, ax2 = plt.subplots()
    cleaned_df['rating'].hist(bins=20, ax=ax2, color='skyblue', edgecolor='black')
    ax2.set_xlabel("Rating")
