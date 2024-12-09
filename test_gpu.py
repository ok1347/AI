from transformers import pipeline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses all logs except errors

# Use PyTorch version of distilGPT2
generator = pipeline('text-generation', model='gpt2', framework='pt')

prompt = "Hello, how are you?"
result = generator(prompt, max_length=50)

# Print only the generated text (answer)
print(result[0]['generated_text'].strip())
