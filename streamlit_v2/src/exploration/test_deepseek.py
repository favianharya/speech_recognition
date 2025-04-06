import torch
from transformers import pipeline
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


class Chatbot:
    def __init__(self, model_name, task, device):
        self.model_name = model_name
        self.task = task
        self.device = device

    def _load_model(self,torch_dtype: str = "auto") -> None:
        """
        load model to the code
        """
        pipe = pipeline(
            self.task,
            model=self.model_name,
            device=self.device,
            torch_dtype=torch_dtype 
        )
        return pipe
    
    def generate_chat_bot(self):
        """
        generate chatbot
        """
        self._load_model()

        # chat history
        chat_history = []
        # system prompt
        system_prompt = "You are a helpful, polite assistant."

        print("Chatbot is ready! Type 'exit' to quit.\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break

            # user input task
            chat_history.append(f"[User]: {user_input}")
            prompt = f"[System]: {system_prompt}\n" + "\n".join(chat_history) + "\n[Assistant]:"

            # generate output
            output = self._load_model(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
            response = output[0]['generated_text'].split("[Assistant]:")[-1].strip()
            
            # LLM response
            chat_history.append(f"[Assistant]: {response}")
            print(f"Bot: {response}\n")

def main():
    chatbot = Chatbot(
        task="text-generation",
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device="cpu"
    )
    chatbot.generate_chat_bot()
    

if __name__ == "__main__":
    main()
