# DermaCare Chatbot for Skin Disease Diagnosis

## Overview
DermaCare Chatbot Doctor is an AI-powered healthcare chatbot designed to assist users in diagnosing skin diseases based on symptoms, descriptions, and uploaded skin images. The chatbot leverages Large Language Models (LLM), Natural Language Processing (NLP), Predictive Modelling (YOLOv5), and OpenCV to deliver accurate and user-friendly responses.

## Features
- **Text-based Diagnosis**: Users can describe their symptoms in natural language and receive possible diagnoses.
- **Image-based Diagnosis**: Users can upload skin images for analysis using the YOLOv5 model.
- **Sentiment Analysis**: Detects user sentiment to provide empathetic responses.
- **Named Entity Recognition (NER)**: Identifies medical terms related to symptoms and diseases.
- **Speech Output**: Uses Google Text-to-Speech (gTTS) to convert responses into audio.

## Key Technologies & Libraries
| Library | Purpose |
|---------|---------|
| `gradio` | User-friendly interface for chatbot interaction |
| `groq` | API client for processing queries with LLM |
| `gtts` | Converts chatbot responses into speech audio |
| `uuid` | Generates unique identifiers for session management |
| `base64 & io.BytesIO` | Encodes and decodes image data |
| `os` | Handles environment variables (e.g., API keys) |
| `logging` | Logs runtime activities for debugging |
| `spacy` | Named Entity Recognition (NER) for medical terms |
| `transformers` | Sentiment analysis using pre-trained NLP models |
| `torch & torchvision` | Power YOLOv5 for skin disease image recognition |
| `PIL` | Image loading and manipulation |
| `cv2 (OpenCV)` | Image preprocessing (resizing, format conversion) |
| `numpy` | Converts image data to NumPy arrays |
| `Ultralytics` | Provides high-performance YOLOv5 models |
| `pandas` | Handles structured data (e.g., CSV files) |
| `scikit-learn` | Implements machine learning models and evaluation metrics |

## Project Modules
### **Module 1: LLM (Large Language Models)**
- **Description**: Captures symptom descriptions and queries in natural language to generate initial diagnoses.
- **Technologies**: Gradio, Groq API
- **Inputs**: User-generated text describing symptoms
- **Outputs**: Possible diagnoses and general advice

### **Module 2: NLP (Natural Language Processing)**
- **Description**: Extracts meaningful insights from user input, performs sentiment analysis, and enhances chatbot responses.
- **Technologies**: spaCy, Transformers
- **Inputs**: Text input from users
- **Outputs**: Extracted medical entities, sentiment analysis results

### **Module 3: Predictive Modelling (YOLOv5)**
- **Description**: Uses a trained YOLOv5 model to classify uploaded skin images into three categories: Atopic Dermatitis, Acne, and Basal Cell Carcinoma (BCC).
- **Technologies**: YOLOv5, Ultralytics, Torch
- **Inputs**: Uploaded skin images
- **Outputs**: Predicted skin disease, confidence score, disease description

### **Module 4: OpenCV (Image Processing)**
- **Description**: Enhances image quality and ensures compatibility with the YOLOv5 model for accurate predictions.
- **Technologies**: OpenCV, NumPy, PIL
- **Inputs**: Uploaded skin images
- **Outputs**: Preprocessed images ready for prediction



## Installation & Setup
### **Prerequisites**
- Python 3.8+
- Virtual environment setup (optional but recommended)

### **Installation Steps**
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/dermacare-chatbot.git
   cd dermacare-chatbot
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set environment variables:
   ```sh
    GROQ_API_KEY_1=your_api_key
   ```
4. Run the chatbot:
   ```sh
   python app.py
   ```

## Usage
1. **Text-based Diagnosis:**
   - Open the chatbot UI (Gradio interface).
   - Enter symptoms (e.g., "I have red patches on my arms").
   - Receive possible diagnoses and advice.
2. **Image-based Diagnosis:**
   - Upload a skin image.
   - The chatbot will analyze and classify the condition.
   - View diagnosis and confidence score.
3. **Audio Response:**
   - The chatbot can convert text responses to speech using gTTS.

## Future Enhancements
- Expand the disease classification dataset.
- Implement multilingual support.
- Improve chatbot accuracy with advanced NLP models.

## Conclusion
DermaCare Chatbot Doctor is a powerful AI-driven tool for preliminary skin disease diagnosis. By integrating state-of-the-art NLP, image processing, and predictive modeling techniques, it provides users with a fast, accessible, and user-friendly healthcare solution.

---
**Author:** RESHMA R B 
**Contact:** reshmarb8547@gmail.com  
**License:** MIT

