import streamlit as st
import ollama

st.title("🎮 Game Review Generator")

name = st.text_input("Game Name", "Battlefield Fury")
price = st.number_input("Price ($)", 0.0, 100.0, 29.99)
tags = st.text_input("Tags (comma-separated)", "Action, Multiplayer, FPS")
platforms = st.multiselect("Platforms", ["Windows", "Mac", "Linux"], ["Windows"])
ccu = st.slider("Peak Concurrent Users", 0, 100000, 5000)
positive = st.slider("Positive Review %", 0, 100, 87)
playtime = st.slider("Avg Playtime (hours)", 0, 200, 25)

if st.button("📝 Generate Review"):
    prompt = f"""
    Generate a game review based on the following metadata:

    - Game Name: {name}
    - Price: ${price}
    - Tags: {tags}
    - Platforms: {', '.join(platforms)}
    - Peak Concurrent Users: {ccu}
    - Positive Review %: {positive}
    - Avg Playtime: {playtime} hours

    Write the review in a casual, friendly tone, as if a Steam user is reviewing it.
    """

    response = ollama.chat(
        model='llama2',
        messages=[{"role": "user", "content": prompt}]
    )
    st.markdown("### 💬 Generated Review:")
    st.write(response['message']['content'])
