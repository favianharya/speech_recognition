import torch
from transformers import pipeline
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

def main():
    pipe = pipeline(
        "text-generation",
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device="cpu",
        torch_dtype="auto"  # or "torch.bfloat16" if your CPU supports it
    )

    # Initialize chat history
    chat_history = []
    system_prompt = "You are a helpful, polite assistant."

    print("Chatbot is ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Append user input to chat history
        chat_history.append(f"[User]: {user_input}")

        # Construct full prompt
        prompt = f"[System]: {system_prompt}\n" + "\n".join(chat_history) + "\n[Assistant]:"

        # Generate response
        output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
        response = output[0]['generated_text'].split("[Assistant]:")[-1].strip()

        # Append assistant response to chat history
        chat_history.append(f"[Assistant]: {response}")

        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()
