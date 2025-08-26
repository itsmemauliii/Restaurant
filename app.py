import streamlit as st
import pandas as pd
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Safe download for Hugging Face (runs once)
nltk.download('punkt')
nltk.download('stopwords')
# 🎨 Set page config with restaurant theme
st.set_page_config(page_title="Restaurant NLP Tool", page_icon="🍛", layout="centered")

# 🏠 App title and description
st.title("🍽️ Restaurant NLP Analyzer")
st.markdown("Upload your restaurant dataset to explore dishes and get mood-based food suggestions.")

# 📤 File uploader
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# 🧠 NLP processing: clean and tokenize text
def process_text_column(df, column):
    text = " ".join(df[column].dropna().astype(str))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    return " ".join(tokens)

# 🌥️ Generate and display word cloud
def show_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# 💖 Suggest dishes based on mood
def mood_matcher(mood):
    mood_map = {
        "happy": ["ice cream", "samosa", "pav bhaji"],
        "sad": ["chocolate", "dal khichdi", "comfort curry"],
        "chill": ["paneer tikka", "wine risotto", "gulab jamun"],
        "lazy": ["instant noodles", "sandwich", "ready-to-eat biryani"],
        "energetic": ["fruit salad", "protein bowl", "green smoothie"]
    }
    return mood_map.get(mood.lower(), [])

# 📊 Main app logic
if uploaded_file:
    try:
        # Read uploaded file
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("Preview of your data:")
        st.dataframe(df.head())

        # Let user select a text column
        text_cols = df.select_dtypes(include='object').columns.tolist()
        selected_col = st.selectbox("Select a column to analyze", text_cols)

        # Show word cloud from selected column
        cleaned_text = process_text_column(df, selected_col)
        st.subheader("🔍 Word Cloud of Selected Column")
        show_wordcloud(cleaned_text)

        # Mood-based food suggestions
        st.subheader("💖 Match Food to Your Mood")
        mood = st.text_input("Enter your mood (e.g., happy, sad, chill)")
        if mood:
            suggestions = mood_matcher(mood)
            if suggestions:
                st.success(f"Based on your mood '{mood}', try:")
                for dish in suggestions:
                    st.markdown(f"- 🍴 {dish}")
            else:
                st.warning("Mood not recognized. Try 'happy', 'sad', 'lazy', etc.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a dataset to begin.")
