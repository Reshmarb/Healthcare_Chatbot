from groq import Groq
import gradio as gr
from gtts import gTTS
import uuid
import base64
from io import BytesIO
import os
import logging
import spacy
from transformers import pipeline
import torch
from PIL import Image
from torchvision import transforms
import pathlib
import cv2  # Import OpenCV
import numpy as np


# Pathlib adjustment for Windows compatibility
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('chatbot_log.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Initialize Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY_1"))
logger.info("Groq client initialized.")

#client = Groq(api_key="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")


# Initialize spaCy NLP model for named entity recognition (NER)
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
logger.info("spaCy NLP model loaded.")

# Initialize sentiment analysis model using Hugging Face
sentiment_analyzer = pipeline("sentiment-analysis")
logger.info("Sentiment analysis model loaded.")

# Load pre-trained YOLOv5 model
def load_yolov5_model():
    logger.info("Loading YOLOv5 model...")
    model = torch.hub.load(r"ultralytics/yolov5", 'custom', path=r'./models/best.pt')
    model.eval()
    logger.info("YOLOv5 model loaded and set to evaluation mode.")
    return model

model = load_yolov5_model()
if model is None:
    logger.error("Failed to load YOLOv5 model.")
    raise RuntimeError("YOLOv5 model loading failed.")


# Function to preprocess user input for better NLP understanding
def preprocess_input(user_input):
    logger.info("Preprocessing user input...")
    user_input = user_input.strip().lower()
    logger.info(f"Preprocessed input: {user_input}")
    return user_input

# Function for sentiment analysis 
def analyze_sentiment(user_input):
    logger.info("Analyzing sentiment...")
    result = sentiment_analyzer(user_input)
    logger.info(f"Sentiment analysis result: {result[0]['label']}")
    return result[0]['label']

# Function to extract medical entities from input using NER
symptoms = [
    "itching", "rash", "dry skin", "redness", "flaking", "cracking skin",
    "swelling", "blisters", "bumps", "peeling skin", "skin discoloration",
    "oozing", "burning sensation", "painful skin", "sensitivity to touch",
    "scaling", "scarring", "ulcers", "bruising", "hives",
    "skin thickening", "sunburn", "lesions", "warts", "pustules",
    "hair loss", "skin darkening", "light patches", "acne", "itchy scalp",
    "stretch marks", "skin irritation", "pigmentation", "eczema flares",
    "seborrhea", "crusty patches", "bleeding skin", "nail discoloration",
    "nail ridges", "skin infections", "skin fissures", "heat rash",
    "papules", "petechiae", "skin tags"
]

diseases = [
    "eczema", "psoriasis", "acne", "rosacea", "dermatitis", "melanoma",
    "basal cell carcinoma", "squamous cell carcinoma", "skin cancer",
    "seborrheic dermatitis", "contact dermatitis", "urticaria",
    "vitiligo", "alopecia areata", "tinea", "ringworm", "onychomycosis",
    "fungal infections", "hyperpigmentation", "hypopigmentation",
    "lichen planus", "cellulitis", "abscess", "boils", "impetigo",
    "keratosis pilaris", "actinic keratosis", "sunburn", "cold sores",
    "herpes simplex", "shingles", "hives", "pityriasis rosea",
    "molluscum contagiosum", "scabies", "lupus rash", "drug eruption",
    "dermatomyositis", "cutaneous lupus", "hidradenitis suppurativa",
    "sebaceous cysts", "keloids", "skin ulcers", "necrotizing fasciitis",
    "warts", "skin infections", "skin abscess", "nail psoriasis"
]


def extract_medical_entities(user_input):
    logger.info("Extracting medical entities...")
    user_input = preprocess_input(user_input)
    medical_entities = []
    for word in user_input.split():
        if word in symptoms or word in diseases:
            medical_entities.append(word)
    logger.info(f"Extracted medical entities: {medical_entities}")
    # return medical_entities

# Function to encode the image
def encode_image(uploaded_image):
    try:
        logger.info("Encoding image...")
        buffered = BytesIO()
        uploaded_image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        logger.info("Image encoding complete.")
        return encoded_image
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        raise


# Initialize messages
def initialize_messages():
    return [{"role": "system",
             "content": '''You are Dr. SkinCare, a highly experienced and professional virtual dermatologist chatbot with over 40 years of expertise in diagnosing and managing skin conditions. You provide accurate, empathetic, and actionable advice on skin-related concerns, including rashes, acne, infections, and chronic skin diseases. Your goal is to offer clear explanations, practical solutions, and guidance on when to seek in-person care from a dermatologist.

You only respond to skin-related inquiries and strive to provide the best possible guidance. Your responses should include:
1. A clear explanation of the possible condition(s) based on the symptoms described.
2. Practical, actionable steps for managing the issue at home (if applicable).
3. A recommendation to consult a dermatologist for persistent, severe, or unclear symptoms.

Maintain a friendly, professional, and empathetic tone in all interactions. Always emphasize that you are not a substitute for professional medical advice, diagnosis, or treatment.

**Prompt Template:**
- Input: Patient’s skin-related concerns, including symptoms, questions, or specific issues they mention.
- Response: Start with a polite acknowledgment of the patient’s concern. Provide a clear, concise explanation of the possible condition(s) and suggest practical, actionable steps. If needed, advise on when to consult a dermatologist.

**Examples:**

- User: "I have a red, itchy rash on my arm. What could it be?"
  Response: "Hello! A red, itchy rash on your arm could be caused by several conditions, such as eczema, contact dermatitis, or a fungal infection. Try keeping the area clean and moisturized, and avoid scratching. Over-the-counter hydrocortisone cream or antihistamines may help. If the rash persists or worsens, please consult a dermatologist for a proper diagnosis."

- User: "I have small, raised bumps on my face. What might this be?"
  Response: "Hi there! Small, raised bumps on your face could be due to acne, closed comedones, or even a mild allergic reaction. Avoid using harsh skincare products and consider using a gentle cleanser and non-comedogenic moisturizer. If the bumps don’t improve or spread, it’s best to see a dermatologist for further evaluation."

- User: "I’ve noticed dark patches on my skin. What could cause this?"
  Response: "Hello! Dark patches on the skin, known as hyperpigmentation, can be caused by sun exposure, hormonal changes, or conditions like melasma. Use a broad-spectrum sunscreen daily and consider skincare products with ingredients like vitamin C or niacinamide. For persistent or concerning patches, consult a dermatologist to rule out underlying conditions."

- User: "I have a mole that looks irregular. Should I be worried?"
  Response: "Hi. Irregular moles can sometimes be a sign of skin changes that need attention. Keep an eye on the mole for changes in size, shape, or color, and avoid exposing it to excessive sunlight. It’s important to have it checked by a dermatologist to rule out any serious concerns, such as skin cancer."

Always maintain a compassionate tone, provide educational insights, and stress that you are not a substitute for professional medical advice. Encourage users to consult a dermatologist for any serious, persistent, or unclear skin concerns.'''
    }]
    logger.info("Messages initialized.")
    return messages

messages = initialize_messages()

# Function for image prediction using YOLOv5
def predict_image(image):
    try:
        logger.info("Predicting image...")
        if image is None:
            logger.error("No image uploaded.")
            return "Error: No image uploaded.", "No description available."

        # Convert and preprocess image
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_resized = cv2.resize(image_np, (224, 224))

        # Transform image
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        im = transform(image_resized).unsqueeze(0)

        # Make predictions
        with torch.no_grad():
            output = model(im)

        # Post-process results
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(output)

        predicted_class_id = torch.argmax(probs, dim=1).item()
        confidence_score = probs[0, predicted_class_id].item()

        if hasattr(model, 'names'):
            class_name = model.names[predicted_class_id]
            prediction_result = f"Predicted Class: {class_name}\nConfidence Score: {confidence_score:.4f}"
            description = get_description(class_name)
        else:
            prediction_result = f"Predicted Class ID: {predicted_class_id}\nConfidence: {confidence_score:.4f}"
            description = "No description available."

        logger.info(f"Prediction result: {prediction_result}")
        return prediction_result, description

    except Exception as e:
        logger.error(f"Error in image prediction: {e}")
        return f"An error occurred during image prediction: {e}", "No description available."

# Function to get description based on predicted class
def get_description(class_name):
    logger.info(f"Getting description for class: {class_name}")
    descriptions = {
        "bcc": "Basal cell carcinoma (BCC) is a type of skin cancer that begins in the basal cells. It often appears as a slightly transparent bump on the skin, though it can take other forms. BCC grows slowly and is unlikely to spread to other parts of the body, but early treatment is important to prevent damage to surrounding tissues.",
        "atopic": "Atopic dermatitis is a chronic skin condition characterized by itchy, inflamed skin. It is common in individuals with a family history of allergies or asthma.",
        "acne": "Acne is a skin condition that occurs when hair follicles become clogged with oil and dead skin cells. It often causes pimples, blackheads, and whiteheads, and is most common among teenagers.",
        # Add more descriptions as needed
    }
    description = descriptions.get(class_name.lower(), "No description available.")
    logger.info(f"Description: {description}")
    return description

# Custom LLM Bot Function
def customLLMBot(user_input, uploaded_image, chat_history):
    try:
        global messages
        logger.info("Processing input...")

        user_input = preprocess_input(user_input)
        sentiment = analyze_sentiment(user_input)
        logger.info(f"Sentiment detected: {sentiment}")

        medical_entities = extract_medical_entities(user_input)
        logger.info(f"Extracted medical entities: {medical_entities}")

        chat_history.append(("user", user_input))

        if uploaded_image is not None:
            base64_image = encode_image(uploaded_image)
            logger.debug(f"Image received, size: {len(base64_image)} bytes")

            messages_image = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ]

            logger.info("Sending image to Groq API for processing...")
            response = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=messages_image,
            )
            logger.info("Image processed successfully.")
        else:
            logger.info("Processing text input...")
            messages.append({
                "role": "user",
                "content": user_input
            })
            response = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=messages,
            )
            logger.info("Text processed successfully.")

        LLM_reply = response.choices[0].message.content
        logger.debug(f"LLM reply: {LLM_reply}")

        chat_history.append(("bot", LLM_reply))
        messages.append({"role": "assistant", "content": LLM_reply})

        audio_file = f"response_{uuid.uuid4().hex}.mp3"
        tts = gTTS(LLM_reply, lang='en')
        tts.save(audio_file)
        logger.info(f"Audio response saved as {audio_file}")

        
        return chat_history, audio_file

    except Exception as e:
        logger.error(f"Error in customLLMBot function: {e}")
        return [("user", user_input or "Image uploaded"), ("bot", f"An error occurred: {e}")], None

# Gradio Interface
def chatbot_ui():
    logger.info("Setting up Gradio interface...")
    with gr.Blocks() as demo:
        gr.Markdown("## Dr. SkinCare - Virtual Dermatologist")

        with gr.Tabs():
        # Chatbot Tab
            with gr.Tab("Chatbot"):
                chat_history = gr.State([])

                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(label="Responses", elem_id="chatbot")
                        user_input = gr.Textbox(
                            label="Ask a health-related question",
                            placeholder="Describe your symptoms...",
                            elem_id="user-input",
                            lines=1,
                        )
                    with gr.Column(scale=1):
                        uploaded_image = gr.Image(label="Upload an Image", type="pil")
                        submit_btn = gr.Button("Submit")
                        clear_btn = gr.Button("Clear")
                        audio_output = gr.Audio(label="Audio Response")

                def handle_submit(user_query, image, history):
                    logger.info("User submitted a query.")
                    response, audio = customLLMBot(user_query, image, history)
                    return response, audio, None, "", history

                user_input.submit(
                    handle_submit,
                    inputs=[user_input, uploaded_image, chat_history],
                    outputs=[chatbot, audio_output, uploaded_image, user_input, chat_history],
                )

                submit_btn.click(
                    handle_submit,
                    inputs=[user_input, uploaded_image, chat_history],
                    outputs=[chatbot, audio_output, uploaded_image, user_input, chat_history],
                )

                clear_btn.click(
                    lambda: ([], "", None, []),
                    inputs=[],
                    outputs=[chatbot, user_input, uploaded_image, chat_history],
                )

        # Predictive Modeling Tab
            with gr.Tab("Predictive Modeling"):
                gr.Markdown("### Upload Image for Prediction")

                with gr.Row():
                    with gr.Column():
                        prediction_image = gr.Image(label="Upload Image", type="pil")
                        predict_btn = gr.Button("Predict")

                    with gr.Column():
                        gr.Markdown("### Prediction Result")
                        prediction_output = gr.Textbox(label="Result", interactive=False)
                        gr.Markdown("### Description")
                        description_output = gr.Textbox(label="Description", interactive=False)
                        clear_prediction_btn = gr.Button("Clear Prediction")

                def clear_prediction(prediction_image, prediction_output, description_output):
                    logger.info("Clearing prediction results.")
                    return None, "", ""

                predict_btn.click(
                    predict_image,
                    inputs=[prediction_image],
                    outputs=[prediction_output, description_output],
                )

                clear_prediction_btn.click(
                    clear_prediction,
                    inputs=[prediction_image, prediction_output, description_output],
                    outputs=[prediction_image, prediction_output, description_output],
                )

    


    logger.info("Gradio interface setup complete.")
    return demo

# Launch the interface
logger.info("Launching chatbot interface...")
chatbot_ui().launch(server_name="0.0.0.0", server_port=7860)

#chatbot_ui().launch(server_name="localhost", server_port=7860)