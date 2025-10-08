# app.py - AI Text Generator with Sentiment Control
import streamlit as st
from transformers import pipeline
import torch

# Configure the page
st.set_page_config(
    page_title="AI Text Generator",
    page_icon="✍️",
    layout="centered"
)

# Title and description
st.title("✍️ AI Text Generator with Sentiment Control")
st.markdown("Generate text based on your input with automatic or manual sentiment selection.")

# Initialize session state for models
@st.cache_resource
def load_sentiment_model():
    """Load the sentiment analysis model"""
    return pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

@st.cache_resource  
def load_text_generator():
    """Load the text generation model"""
    return pipeline("text-generation", model="gpt2", max_length=300, torch_dtype=torch.bfloat16)

# Load models
with st.spinner("Loading AI models..."):
    sentiment_classifier = load_sentiment_model()
    text_generator = load_text_generator()

# User input section
st.header("📝 Input Settings")

# Text input
user_prompt = st.text_area(
    "Enter your prompt:",
    placeholder="e.g., The future of artificial intelligence in healthcare...",
    height=100
)

# Layout for sentiment controls
col1, col2 = st.columns(2)

with col1:
    # Sentiment mode selection
    sentiment_mode = st.radio(
        "Sentiment Control:",
        ["Auto-detect", "Manual selection"],
        help="Choose whether to auto-detect sentiment or select it manually"
    )

with col2:
    # Manual sentiment selection
    if sentiment_mode == "Manual selection":
        selected_sentiment = st.selectbox(
            "Choose sentiment:",
            ["POSITIVE", "NEGATIVE", "NEUTRAL"],
            index=0
        )
    else:
        selected_sentiment = None

# Generation settings
st.subheader("⚙️ Generation Settings")

col3, col4 = st.columns(2)

with col3:
    max_length = st.slider(
        "Maximum length:",
        min_value=50,
        max_value=300,
        value=150,
        step=30,
        help="Maximum number of words in generated text"
    )

with col4:
    temperature = st.slider(
        "Creativity (temperature):",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more creative, lower values more focused"
    )

# Generate button
generate_btn = st.button(
    "✨ Generate Text",
    type="primary",
    use_container_width=True
)

# Main generation function
def generate_text_with_sentiment(prompt, sentiment_mode, manual_sentiment=None):
    """Generate text based on prompt and sentiment settings"""
    
    # Determine sentiment
    if sentiment_mode == "Auto-detect" and prompt.strip():
        # Auto-detect sentiment
        sentiment_result = sentiment_classifier(prompt)[0]
        detected_sentiment = sentiment_result['label']
        confidence = sentiment_result['score']
        
        st.success(f"🎭 Detected sentiment: **{detected_sentiment}** (confidence: {confidence:.2f})")
        final_sentiment = detected_sentiment
    elif sentiment_mode == "Manual selection":
        final_sentiment = manual_sentiment
        st.info(f"🎭 Using manually selected sentiment: **{final_sentiment}**")
    else:
        final_sentiment = "NEUTRAL"
        st.info("ℹ️ Using default NEUTRAL sentiment")
    
    # Create enhanced prompt with sentiment guidance
    sentiment_instructions = {
        "POSITIVE": f"Write a positive, enthusiastic, and optimistic text about: {prompt}",
        "NEGATIVE": f"Write a negative, critical, or concerned text about: {prompt}", 
        "NEUTRAL": f"Write a balanced, factual, and neutral text about: {prompt}"
    }
    
    enhanced_prompt = sentiment_instructions.get(final_sentiment, prompt)
    
    # Generate text
    with st.spinner(f"Generating {final_sentiment.lower()} text..."):
        try:
            # Update generator parameters
            text_generator.max_length = max_length
            generation_args = {
                "max_length": max_length,
                "truncation": True,
                "num_return_sequences": 1,
                "temperature": temperature,
                "repetition_penalty": 1.2,
                "do_sample": True,
                "pad_token_id": text_generator.tokenizer.eos_token_id
            }
            
            result = text_generator(enhanced_prompt, **generation_args)
            generated_text = result[0]['generated_text']
            
            # Clean up the output by removing the original prompt if it's repeated
            if generated_text.startswith(enhanced_prompt):
                generated_text = generated_text[len(enhanced_prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            st.error(f"Error during text generation: {str(e)}")
            return None

# Main application logic
if generate_btn:
    if not user_prompt.strip():
        st.warning("⚠️ Please enter a prompt to generate text.")
    else:
        # Generate text
        generated_text = generate_text_with_sentiment(
            user_prompt, 
            sentiment_mode, 
            selected_sentiment
        )
        
        # Display results
        if generated_text:
            st.header("📄 Generated Text")
            st.text_area(
                "Generated Text:",
                generated_text,
                height=200,
                label_visibility="collapsed"
            )
            
            # Add copy and download options
            col5, col6 = st.columns(2)
            with col5:
                st.code(generated_text)
            with col6:
                st.download_button(
                    label="📥 Download Text",
                    data=generated_text,
                    file_name="generated_text.txt",
                    mime="text/plain"
                )

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This AI Text Generator uses:
    - **Sentiment Analysis**: BERT-based model for emotion detection
    - **Text Generation**: GPT-2 for creative text generation
    - **Streamlit**: For the interactive web interface
    
    ### How to use:
    1. Enter your text prompt
    2. Choose auto-detect or manual sentiment
    3. Adjust generation settings
    4. Click 'Generate Text'
    """)
    
    st.header("🔧 Models Used")
    st.markdown("""
    - **Sentiment**: `bertweet-base-sentiment-analysis`
    - **Text Generation**: `gpt2`
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit & Hugging Face Transformers 🤗")