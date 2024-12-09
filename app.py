from flask import Flask, request, render_template, jsonify
import whisper
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import time
import logging
import warnings

# Log in programmatically
login(token="your_huggingface_access_token")

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)

# GPU or CPU device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Running on device: {device}")

# Load the Whisper model for transcription
whisper_model = whisper.load_model("base", device=device)

# Load the LLaMA model (e.g., LLaMA 2 chat)
llama_model_name = "tiiuae/falcon-7b-instruct"  # Replace with your desired LLaMA model path or name
tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure padding token is set
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)

# Create a pipeline for text generation without specifying the device
llm_pipeline = pipeline("text-generation", model=llama_model, tokenizer=tokenizer)

# Configure folder for storing uploaded audio files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    if request.method == "POST":
        if "audio" not in request.files:
            return render_template("index.html", response="No audio file received.")
        
        audio_file = request.files["audio"]
        filename = f"recording_{int(time.time())}.webm"  # Save with unique filenames
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        audio_file.save(file_path)
        logger.info(f"Audio file saved: {filename}")

        # Transcribe the audio file using Whisper
        try:
            result = whisper_model.transcribe(file_path)
            transcription = result["text"]
            logger.info(f"Transcription: {transcription}")

            # Tokenize transcription and generate a response
            inputs = tokenizer(
                transcription,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=200
            )
            outputs = llama_model.generate(**inputs, max_length=200)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return render_template("index.html", response="An error occurred during processing.")
    return render_template("index.html", response=response)


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio_data" not in request.files:
        return jsonify({"error": "No audio file received."}), 400

    # Save the uploaded audio file
    audio_file = request.files["audio_data"]
    filename = f"recording_{int(time.time())}.webm"  # Save each recording uniquely
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    audio_file.save(file_path)
    logger.info(f"Saved audio file: {filename}")

    try:
        # Transcribe the audio using Whisper
        result = whisper_model.transcribe(file_path)
        transcription = result["text"]
        logger.info(f"Transcription: {transcription}")

        # Tokenize transcription and generate a response
        inputs = tokenizer(
            transcription,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=200
        )
        outputs = llama_model.generate(**inputs, max_length=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"transcription": transcription, "response": response})
    except Exception as e:
        logger.error(f"Error during transcription or generation: {e}")
        return jsonify({"error": "An error occurred during processing."}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)


