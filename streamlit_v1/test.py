from huggingface_hub import login
from transformers import pipeline

login(token="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")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Define your prompt
prompt = "How many r in strawberry?"
messages = [{"role": "user", "content": prompt}]

# Tokenizing the message
tokenized_message = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)

# Generating the response
response_token_ids = model.generate(tokenized_message['input_ids'], 
                                     attention_mask=tokenized_message['attention_mask'],  
                                     max_new_tokens=4096, 
                                     pad_token_id=tokenizer.eos_token_id)

# Decoding the generated tokens
generated_tokens = response_token_ids[:, len(tokenized_message['input_ids'][0]):]
generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

print(generated_text)

