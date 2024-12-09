import whisper

model = whisper.load_model("base")
result = model.transcribe("test_audio.mp3")
print(result["text"])
llama_model_name = "mistralai/Mistral-7B"
llama_model_name = "EleutherAI/gpt-j-6B"
llama_model_name = "EleutherAI/pythia-6.9b"