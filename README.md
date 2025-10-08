Sentiment-Aware AI Text Generator
An interactive web application that combines sentiment analysis and text generation to produce emotionally aligned content. Users can input any prompt and choose between auto-detected or manually selected sentiments (Positive/Negative/Neutral) to guide the AI's creative output.

Features:
🤖 Real-time sentiment analysis using BERT-based models
✍️ GPT-2 powered text generation
🎛️ Manual sentiment override options
⚙️ Adjustable generation parameters (length, creativity)
🎨 Streamlit-based interactive interface

🚧 Challenges & Limitations
1. Technical Challenges ->
-> Model Performance & Memory Constraints :
Large Model Sizes: Pre-trained transformer models require significant RAM (1-2GB+), limiting deployment options on free-tier platforms
Slow Initial Loading: Models download and load into memory on first run, causing 30-60 second startup delays
VRAM Requirements: GPU memory constraints affect performance on consumer hardware

-> Text Generation Quality
Repetition Issues: GPT-2 tends to repeat phrases and sentences, especially with conservative temperature settings
Context Limitation: Maximum token limits (512 for many models) restrict long-form content generation
Sentiment Alignment: Generated text doesn't always perfectly match the intended emotional tone

-> Pipeline Integration
Tokenization Warnings: Required explicit truncation=True parameters to handle varying input lengths
Parameter Tuning: Finding optimal temperature, repetition penalty, and length parameters required extensive experimentation
Stream Compatibility: Ensuring smooth data flow between sentiment analysis and text generation pipelines

2. Performance Limitations ->
-> Generation Speed
Real-time Delays: Text generation takes 5-15 seconds depending on length and complexity
Cold Start Times: Initial model loading can take up to a minute on slower systems
Concurrent Users: Streamlit's architecture may struggle with multiple simultaneous users

-> Accuracy Constraints
Sentiment Misclassification: BERT-based models can misinterpret nuanced or sarcastic language
Content Coherence: Generated text may lack logical flow or context consistency
Emotional Nuance: Difficulty capturing subtle emotional differences (e.g., joyful vs. content)

-> Model Capabilities
Single Language: Currently supports only English text generation
Content Restrictions: Limited ability to handle specialized domains (technical, medical, legal)
No Multi-turn Conversations: Cannot maintain context across multiple interactions

