from locust import HttpUser, task, between
import random, json

PROMPTS = [
    "Write a short description of the Eiffel Tower.",
    "Summarize the following text: AI can transform industries by automating tasks...",
    "What is the capital of France?",
    "Explain the effect of caffeine on sleep in 3 sentences.",
    "Provide a brief cookbook recipe for pancakes."
]

class SpeculativeUser(HttpUser):
    wait_time = between(0.5, 2.0)
    @task
    def speculative_infer(self):
        prompt = random.choice(PROMPTS)
        payload = {"prompt": prompt, "max_tokens": 64, "risk_level": "default", "tenant": "locust-test"}
        self.client.post("/infer_speculative", json=payload, name="/infer_speculative")
