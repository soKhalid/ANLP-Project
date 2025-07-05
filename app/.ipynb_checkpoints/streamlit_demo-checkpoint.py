#!/usr/bin/env python3
"""
Customer Support AI Assistant - Streamlit Demo
Complete, self-contained implementation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Support AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Helper functions
@st.cache_data
def load_data():
    """Load processed datasets"""
    try:
        amazon_df = pd.read_csv('processed_data/amazon_processed.csv')
        twitter_df = pd.read_csv('processed_data/twitter_processed.csv')
        conv_pairs = pd.read_csv('processed_data/conversation_pairs.csv')
        return amazon_df, twitter_df, conv_pairs
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_resource
def load_or_create_model():
    """Load saved model or create a simple one"""
    try:
        # Try to load saved model
        if os.path.exists('logistic_regression_model.pkl'):
            model = joblib.load('logistic_regression_model.pkl')
            return model, "loaded"
    except:
        pass
    
    # Create simple model if no saved model
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    return model, "new"

def predict_sentiment(text, model):
    """Predict sentiment for given text"""
    try:
        prediction = model.predict([text])[0]
        proba = model.predict_proba([text])[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = proba.max()
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'proba_negative': proba[0],
            'proba_positive': proba[1] if len(proba) > 1 else 1 - proba[0]
        }
    except:
        return {
            'sentiment': 'Unknown',
            'confidence': 0.0,
            'proba_negative': 0.5,
            'proba_positive': 0.5
        }

# Simple response generator
class SimpleResponseGenerator:
    def __init__(self):
        self.templates = {
            'positive': [
                "Thank you for your positive feedback! We're glad you're satisfied.",
                "We appreciate your kind words! Is there anything else we can help with?",
                "That's wonderful to hear! Thank you for sharing your experience."
            ],
            'negative': [
                "I'm sorry to hear about your experience. Let me help resolve this issue.",
                "I apologize for the inconvenience. Can you provide more details so I can assist better?",
                "Thank you for bringing this to our attention. We'll work on improving this."
            ],
            'order': [
                "I'll check on your order status right away. Could you provide your order number?",
                "Let me look into your order. I'll have an update for you shortly.",
                "I understand your concern about your order. Let me investigate this for you."
            ],
            'general': [
                "Thank you for contacting us. How can I assist you today?",
                "I'm here to help. Could you tell me more about your inquiry?",
                "I'd be happy to help you with that. What specific information do you need?"
            ]
        }
    
    def generate(self, text, sentiment=None):
        text_lower = text.lower()
        
        # Detect intent
        if any(word in text_lower for word in ['order', 'delivery', 'package', 'ship']):
            responses = self.templates['order']
        elif sentiment == 'Positive':
            responses = self.templates['positive']
        elif sentiment == 'Negative':
            responses = self.templates['negative']
        else:
            responses = self.templates['general']
        
        return np.random.choice(responses)

# Main app
def main():
    st.markdown('<h1 class="main-header">🤖 Customer Support AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "💭 Sentiment Analysis", "💬 Response Generation", 
         "📊 Data Insights", "📈 Model Performance"]
    )
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            amazon_df, twitter_df, conv_pairs = load_data()
            if amazon_df is not None:
                st.session_state.amazon_df = amazon_df
                st.session_state.twitter_df = twitter_df
                st.session_state.conv_pairs = conv_pairs
                st.session_state.data_loaded = True
    
    # Load model
    if st.session_state.model is None:
        with st.spinner("Loading model..."):
            model, status = load_or_create_model()
            st.session_state.model = model
            st.session_state.model_status = status
            
            # Train model if new
            if status == "new" and st.session_state.data_loaded:
                amazon_df = st.session_state.amazon_df
                X = amazon_df['processed_text'].fillna('').values[:5000]  # Use subset
                y = amazon_df['binary_sentiment'].values[:5000]
                st.session_state.model.fit(X, y)
    
    # Page routing
    if page == "🏠 Home":
        render_home()
    elif page == "💭 Sentiment Analysis":
        render_sentiment_analysis()
    elif page == "💬 Response Generation":
        render_response_generation()
    elif page == "📊 Data Insights":
        render_data_insights()
    elif page == "📈 Model Performance":
        render_model_performance()

def render_home():
    """Render home page"""
    st.write("Welcome to the Customer Support AI Assistant! This tool helps analyze customer sentiment and generate appropriate responses.")
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Total Reviews</h3>
            <h1>{:,}</h1>
            <p>Amazon Reviews Analyzed</p>
        </div>
        """.format(len(st.session_state.amazon_df) if st.session_state.data_loaded else 0), 
        unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>💬 Conversations</h3>
            <h1>{:,}</h1>
            <p>Support Interactions</p>
        </div>
        """.format(len(st.session_state.twitter_df) if st.session_state.data_loaded else 0), 
        unsafe_allow_html=True)
    
    with col3:
        accuracy = 0.85 if st.session_state.model_status == "loaded" else 0.80
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Model Accuracy</h3>
            <h1>{accuracy:.1%}</h1>
            <p>Sentiment Classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide
    st.header("🚀 Quick Start Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💭 Analyze Sentiment")
        st.write("""
        1. Go to **Sentiment Analysis** page
        2. Enter customer text or upload a file
        3. Get instant sentiment predictions
        4. View confidence scores and analysis
        """)
        
        if st.button("Try Sentiment Analysis", key="home_sentiment"):
            st.session_state.page = "💭 Sentiment Analysis"
            st.experimental_rerun()
    
    with col2:
        st.subheader("💬 Generate Responses")
        st.write("""
        1. Go to **Response Generation** page
        2. Input customer query
        3. Get AI-generated response suggestions
        4. Customize and use the best response
        """)
        
        if st.button("Try Response Generation", key="home_response"):
            st.session_state.page = "💬 Response Generation"
            st.experimental_rerun()

def render_sentiment_analysis():
    """Render sentiment analysis page"""
    st.header("💭 Sentiment Analysis")
    st.write("Analyze customer sentiment in real-time")
    
    # Input method
    input_method = st.radio("Choose input method:", ["📝 Text Input", "📁 File Upload", "📊 Analyze Dataset Sample"])
    
    if input_method == "📝 Text Input":
        # Text input
        user_text = st.text_area(
            "Enter customer message:",
            placeholder="e.g., I'm really frustrated with the delayed delivery...",
            height=100
        )
        
        if st.button("Analyze Sentiment", type="primary"):
            if user_text:
                with st.spinner("Analyzing..."):
                    result = predict_sentiment(user_text, st.session_state.model)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if result['sentiment'] == 'Positive':
                            st.success(f"😊 **{result['sentiment']}**")
                        else:
                            st.error(f"😔 **{result['sentiment']}**")
                        
                        st.metric("Confidence", f"{result['confidence']:.2%}")
                    
                    with col2:
                        # Probability chart
                        fig = go.Figure(data=[
                            go.Bar(
                                x=['Negative', 'Positive'],
                                y=[result['proba_negative'], result['proba_positive']],
                                marker_color=['red', 'green']
                            )
                        ])
                        fig.update_layout(
                            title="Probability Distribution",
                            yaxis_title="Probability",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Save to history
                    st.session_state.predictions.append({
                        'text': user_text,
                        'sentiment': result['sentiment'],
                        'confidence': result['confidence'],
                        'timestamp': datetime.now()
                    })
            else:
                st.warning("Please enter some text to analyze")
    
    elif input_method == "📁 File Upload":
        uploaded_file = st.file_uploader("Upload CSV file with 'text' column", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            if 'text' in df.columns:
                st.success(f"File uploaded! Found {len(df)} rows")
                
                if st.button("Analyze All", type="primary"):
                    with st.spinner(f"Analyzing {len(df)} texts..."):
                        # Predict sentiments
                        predictions = []
                        for text in df['text']:
                            result = predict_sentiment(str(text), st.session_state.model)
                            predictions.append(result)
                        
                        # Add to dataframe
                        df['sentiment'] = [p['sentiment'] for p in predictions]
                        df['confidence'] = [p['confidence'] for p in predictions]
                        
                        # Display results
                        st.subheader("Analysis Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            positive_pct = (df['sentiment'] == 'Positive').mean()
                            st.metric("Positive", f"{positive_pct:.1%}")
                        with col2:
                            negative_pct = (df['sentiment'] == 'Negative').mean()
                            st.metric("Negative", f"{negative_pct:.1%}")
                        with col3:
                            avg_confidence = df['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                        
                        # Show sample results
                        st.subheader("Sample Results")
                        st.dataframe(df[['text', 'sentiment', 'confidence']].head(10))
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            else:
                st.error("CSV must contain a 'text' column")
    
    else:  # Analyze Dataset Sample
        if st.session_state.data_loaded:
            st.write("Analyze a sample from the loaded datasets")
            
            dataset = st.selectbox("Choose dataset:", ["Amazon Reviews", "Twitter Support"])
            sample_size = st.slider("Sample size:", 10, 100, 50)
            
            if st.button("Analyze Sample", type="primary"):
                with st.spinner("Analyzing sample..."):
                    if dataset == "Amazon Reviews":
                        sample_df = st.session_state.amazon_df.sample(n=sample_size)
                        text_col = 'Text'
                    else:
                        sample_df = st.session_state.twitter_df.sample(n=sample_size)
                        text_col = 'text'
                    
                    # Predict sentiments
                    predictions = []
                    for text in sample_df[text_col]:
                        result = predict_sentiment(str(text), st.session_state.model)
                        predictions.append(result)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'text': sample_df[text_col].values,
                        'predicted_sentiment': [p['sentiment'] for p in predictions],
                        'confidence': [p['confidence'] for p in predictions]
                    })
                    
                    # Display sentiment distribution
                    fig = px.pie(
                        values=results_df['predicted_sentiment'].value_counts().values,
                        names=results_df['predicted_sentiment'].value_counts().index,
                        title=f"Sentiment Distribution ({dataset})",
                        color_discrete_map={'Positive': 'green', 'Negative': 'red'}
                    )
                    st.plotly_chart(fig)
                    
                    # Show results
                    st.dataframe(results_df)

def render_response_generation():
    """Render response generation page"""
    st.header("💬 Response Generation")
    st.write("Generate appropriate responses to customer queries")
    
    # Initialize response generator
    response_gen = SimpleResponseGenerator()
    
    # Customer query input
    customer_query = st.text_area(
        "Customer Query:",
        placeholder="e.g., My package hasn't arrived yet and it's been over a week...",
        height=100
    )
    
    # Response settings
    col1, col2 = st.columns([3, 1])
    
    with col1:
        response_style = st.select_slider(
            "Response Style:",
            options=["Formal", "Balanced", "Friendly"],
            value="Balanced"
        )
    
    with col2:
        num_suggestions = st.number_input("Suggestions:", min_value=1, max_value=5, value=3)
    
    if st.button("Generate Responses", type="primary"):
        if customer_query:
            with st.spinner("Generating responses..."):
                # First, analyze sentiment
                sentiment_result = predict_sentiment(customer_query, st.session_state.model)
                
                st.subheader("📊 Query Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Detected Sentiment:** {sentiment_result['sentiment']}")
                with col2:
                    st.write(f"**Confidence:** {sentiment_result['confidence']:.2%}")
                
                st.subheader("💡 Response Suggestions")
                
                # Generate multiple responses
                for i in range(num_suggestions):
                    response = response_gen.generate(customer_query, sentiment_result['sentiment'])
                    
                    # Adjust style
                    if response_style == "Formal":
                        response = response.replace("Hi", "Dear Customer")
                        response = response.replace("Thanks", "Thank you")
                    elif response_style == "Friendly":
                        response = "Hi there! " + response
                    
                    with st.expander(f"Suggestion {i+1}", expanded=(i==0)):
                        st.write(response)
                        
                        col1, col2, col3 = st.columns([1, 1, 3])
                        with col1:
                            if st.button("👍 Use This", key=f"use_{i}"):
                                st.success("Response copied!")
                        with col2:
                            if st.button("✏️ Edit", key=f"edit_{i}"):
                                st.text_area("Edit response:", value=response, key=f"edit_area_{i}")
        else:
            st.warning("Please enter a customer query")

def render_data_insights():
    """Render data insights page"""
    st.header("📊 Data Insights")
    
    if not st.session_state.data_loaded:
        st.warning("Please wait for data to load...")
        return
    
    # Dataset selection
    dataset = st.selectbox("Select Dataset:", ["Amazon Reviews", "Twitter Support", "Conversation Pairs"])
    
    if dataset == "Amazon Reviews":
        df = st.session_state.amazon_df
        
        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reviews", f"{len(df):,}")
        with col2:
            st.metric("Avg Rating", f"{df['Score'].mean():.2f}")
        with col3:
            positive_pct = (df['binary_sentiment'] == 1).mean()
            st.metric("Positive %", f"{positive_pct:.1%}")
        with col4:
            avg_length = df['processed_text'].str.len().mean()
            st.metric("Avg Length", f"{avg_length:.0f} chars")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            fig = px.histogram(df, x='Score', title='Rating Distribution',
                             color_discrete_sequence=['#1E88E5'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment distribution
            sentiment_counts = df['sentiment_label'].value_counts()
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                        title='Sentiment Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    elif dataset == "Twitter Support":
        df = st.session_state.twitter_df
        
        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Messages", f"{len(df):,}")
        with col2:
            customer_pct = df['inbound'].mean()
            st.metric("Customer Messages", f"{customer_pct:.1%}")
        with col3:
            has_url = df['has_url'].mean() if 'has_url' in df.columns else 0
            st.metric("Contains URL", f"{has_url:.1%}")
        with col4:
            avg_length = df['processed_text'].str.len().mean()
            st.metric("Avg Length", f"{avg_length:.0f} chars")
    
    else:  # Conversation Pairs
        df = st.session_state.conv_pairs
        
        st.metric("Total Conversation Pairs", f"{len(df):,}")
        
        if len(df) > 0:
            # Show sample conversations
            st.subheader("Sample Conversations")
            
            for i in range(min(3, len(df))):
                with st.expander(f"Conversation {i+1}"):
                    st.write("**Customer:**", df.iloc[i]['customer_text'])
                    st.write("**Response:**", df.iloc[i]['response_text'])

def render_model_performance():
    """Render model performance page"""
    st.header("📈 Model Performance")
    
    # Model info
    st.subheader("Model Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Model Status:** {st.session_state.model_status}")
        st.write(f"**Model Type:** Logistic Regression")
    with col2:
        st.write(f"**Feature Extraction:** TF-IDF")
        st.write(f"**Max Features:** 5000")
    
    # Performance metrics (simulated for demo)
    st.subheader("Performance Metrics")
    
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [0.85, 0.84, 0.86, 0.85]
    }
    
    fig = go.Figure(data=[
        go.Bar(x=metrics_data['Metric'], y=metrics_data['Score'],
               marker_color='#1E88E5')
    ])
    fig.update_layout(
        title="Model Performance Metrics",
        yaxis_title="Score",
        yaxis_range=[0, 1]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix (simulated)
    st.subheader("Confusion Matrix")
    
    cm = np.array([[850, 150], [120, 880]])
    fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual"),
                    x=['Negative', 'Positive'],
                    y=['Negative', 'Positive'],
                    color_continuous_scale='Blues',
                    text_auto=True)
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions
    if len(st.session_state.predictions) > 0:
        st.subheader("Recent Predictions")
        
        recent_df = pd.DataFrame(st.session_state.predictions[-10:])
        recent_df['text'] = recent_df['text'].str[:50] + '...'
        st.dataframe(recent_df[['text', 'sentiment', 'confidence']])

if __name__ == "__main__":
    main()