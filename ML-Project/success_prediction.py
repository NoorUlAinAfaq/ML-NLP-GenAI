import streamlit as st
import joblib
import numpy as np
import os
from pathlib import Path

@st.cache_resource



# Get the current directory

def load_model():
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    model_path = current_dir / "success_model.pkl"
    scaler_path = current_dir / "success_scaler.pkl"
    model = joblib.load(model_path) 
    scaler = joblib.load(scaler_path)  # Assuming you saved the scaler
    return model, scaler

st.title("🎮 Game Success Predictor")

try:
    model, scaler = load_model()
    
    # Define feature list - must match your training data exactly
    model_features = ['price', 'required_age', 'dlc_count', 'windows_numeric', 'mac_numeric',
                      'linux_numeric', 'platform_count', 'years_since_release',
                      'metacritic_score', 'achievements', 'recommendations',
                      'average_playtime_forever', 'median_playtime_forever', 'peak_ccu',
                      'pct_pos_total', 'num_reviews_total', 'revenue_estimate',
                      'price_metacritic_ratio', 'price_per_hour', 'engagement_score',
                      'tag_Action', 'tag_Adventure', 'tag_Singleplayer', 'tag_Casual',
                      'tag_Indie', 'tag_2D', 'tag_Strategy', 'tag_Simulation', 'tag_RPG',
                      'tag_Exploration']
    
    # Collect inputs
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            price = st.number_input("Price ($)", min_value=0.0, max_value=100.0, value=29.99)
            required_age = st.slider("Required Age", 0, 21, 12)
            dlc_count = st.slider("DLC Count", 0, 50, 2)
            metacritic_score = st.slider("Metacritic Score", 0, 100, 75)
            achievements = st.slider("Achievements", 0, 500, 30)
            recommendations = st.slider("User Recommendations", 0, 100000, 5000)
            average_playtime = st.slider("Average Playtime (hrs)", 0, 2000, 150)
            median_playtime = st.slider("Median Playtime (hrs)", 0, 1000, 50)
        
        with col2:
            peak_ccu = st.slider("Peak Concurrent Users", 0, 200000, 5000)
            pos_reviews = st.slider("Positive Reviews %", 0, 100, 85)
            total_reviews = st.slider("Number of Total Reviews", 0, 500000, 10000)
            revenue = st.slider("Estimated Revenue ($)", 0, 2000000, 500000)
            years_since_release = st.slider("Years Since Release", 0.0, 15.0, 2.5)
            
            st.write("Supported Platforms:")
            windows = st.checkbox("Windows", value=True)
            mac = st.checkbox("Mac", value=False)
            linux = st.checkbox("Linux", value=False)
            
        # Calculate derived values
        platform_count = int(windows) + int(mac) + int(linux)
        
        # Calculate derived metrics
        price_metacritic_ratio = round(price / max(metacritic_score, 1), 2)
        price_per_hour = round(price / max(average_playtime, 1), 2)
        engagement_score = round((median_playtime + peak_ccu / 100) / 2, 2)
        
        # Game tags section
        st.subheader("Game Tags")
        tags = ['Action', 'Adventure', 'Singleplayer', 'Casual', 'Indie', 
                '2D', 'Strategy', 'Simulation', 'RPG', 'Exploration']
        
        # Create a 2x5 grid for tags
        tag_cols = st.columns(5)
        tag_values = {}
        
        for i, tag in enumerate(tags):
            with tag_cols[i % 5]:
                tag_values[f'tag_{tag}'] = int(st.checkbox(tag, value=False))
                
        submit_button = st.form_submit_button("Predict Success")
    
    if submit_button:
        # Create a game dictionary with all inputs
        game = {
            'price': price,
            'required_age': required_age,
            'dlc_count': dlc_count,
            'windows_numeric': int(windows),
            'mac_numeric': int(mac),
            'linux_numeric': int(linux),
            'platform_count': platform_count,
            'years_since_release': years_since_release,
            'metacritic_score': metacritic_score,
            'achievements': achievements,
            'recommendations': recommendations,
            'average_playtime_forever': average_playtime,
            'median_playtime_forever': median_playtime,
            'peak_ccu': peak_ccu,
            'pct_pos_total': pos_reviews,
            'num_reviews_total': total_reviews,
            'revenue_estimate': revenue,
            'price_metacritic_ratio': price_metacritic_ratio,
            'price_per_hour': price_per_hour,
            'engagement_score': engagement_score
        }
        
        # Add tag values to the game dictionary
        for tag, value in tag_values.items():
            game[tag] = value
        
        # Create feature vector ensuring correct order
        feature_vector = [game[feat] for feat in model_features]
        
        # Convert to numpy array with proper shape
        features = np.array([feature_vector])
        features_scaled = scaler.transform(features)  # Only transform, don't fit
        prediction = model.predict(features_scaled)[0]
        
        # Display prediction
        st.subheader("Prediction Results")
        
        # Show the numerical prediction
        st.markdown(f"### 🔮 Predicted Success Score: **{prediction:.2f}**")
        
        # Visual indicator
        st.progress(min(max(float(prediction), 0.0), 1.0))  # Ensure in 0-1 range
        
        # Interpretation
        if prediction > 0.75:
            st.success("🎉 High success potential! Your game has strong indicators of commercial success.")
        elif prediction > 0.5:
            st.info("🙂 Moderate success potential. Consider adjusting some parameters to improve chances.")
        else:
            st.warning("⚠️ Low success potential. You might want to reconsider some aspects of your game.")
        
        # Display key metrics that influenced the prediction
        st.subheader("Key Metrics")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        metrics_col1.metric("Price", f"${price:.2f}")
        metrics_col2.metric("Metacritic Score", f"{metacritic_score}")
        metrics_col3.metric("User Reviews", f"{pos_reviews}% Positive")
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        metrics_col1.metric("Price/Hour", f"${price_per_hour:.2f}")
        metrics_col2.metric("Engagement Score", f"{engagement_score:.2f}")
        metrics_col3.metric("Platforms", f"{platform_count}")
        
except Exception as e:
    st.error(f"Error loading model or making prediction: {e}")
    st.write("Please make sure 'success_model.pkl' and 'success_scaler.pkl' are in the same directory as this app.")

# Add explanation in sidebar
st.sidebar.title("About")
st.sidebar.info("""
This app predicts the commercial success potential of a video game based on various features.
Enter the parameters for your game and click 'Predict Success' to see the estimated success score.
""")