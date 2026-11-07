import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from PIL import Image
import requests
import os
import re

try:
    import spacy
    NER = spacy.load("en_core_web_sm")
except ImportError:
    NER = None

logger = logging.getLogger(__name__)

class MultimodalAISystem:
    DEFAULT_MODEL_IDS = {
        "core_llm": "gpt2",
        "vision_model": "Salesforce/blip-image-captioning-base",
        "image_gen": "CompVis/stable-diffusion-v1-4"
    }

    def __init__(self, model_ids=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.tokenizer = None
        self.model_ids = model_ids if model_ids else self.DEFAULT_MODEL_IDS
        self.WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")

    def load_core_llm(self):
        if "core_llm" not in self.models:
            logger.info(f"Loading core LLM model: {self.model_ids['core_llm']} ...")
            self.models["core_llm"] = AutoModelForCausalLM.from_pretrained(
                self.model_ids["core_llm"], torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_ids["core_llm"])

    def load_vision_model(self):
        if "vision_model" not in self.models:
            logger.info(f"Loading vision model: {self.model_ids['vision_model']} ...")
            self.models["vision_model"] = pipeline(
                "image-to-text", model=self.model_ids["vision_model"], device=0 if self.device == "cuda" else -1
            )

    def load_image_generator(self):
        if "image_gen" not in self.models:
            logger.info(f"Loading image generator: {self.model_ids['image_gen']} ...")
            self.models["image_gen"] = pipeline(
                "text-to-image", model=self.model_ids["image_gen"], device=0 if self.device == "cuda" else -1
            )

    def handle_text_query(self, prompt):
        self.load_core_llm()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.models["core_llm"].generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(prompt):].strip()
        return answer

    def handle_image_query(self, image_path):
        self.load_vision_model()
        image = Image.open(image_path)
        result = self.models["vision_model"](image)
        return result[0]['generated_text'] if result else "No description generated."

    def generate_image(self, prompt):
        self.load_image_generator()
        result = self.models["image_gen"](prompt)
        if isinstance(result, list) and result:
            image = result[0]
            output_path = "generated_image.png"
            image.save(output_path)
            return f"Image generated and saved to {output_path}."
        return "Image generation failed."

    def access_real_time_data(self, query):
        city = self.extract_city_from_query(query)
        if not city:
            return "Please specify a city for the weather information."
        if not self.WEATHER_API_KEY:
            return "Weather API key is not set."
        response = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.WEATHER_API_KEY}&units=metric"
        )
        data = response.json()
        if response.status_code != 200 or 'main' not in data:
            return "Failed to retrieve weather data."
        return f"Current temperature in {city} is {data['main']['temp']}°C."

    @staticmethod
    def extract_city_from_query(query):
        if NER:
            doc = NER(query)
            cities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
            if cities:
                return cities[-1]
        match = re.search(r'in\s+([A-Za-z ]+)', query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def generate_safe_response(self, query):
        q_lower = query.lower()
        if "generate an image" in q_lower:
            prompt = query.replace("generate an image", "", 1).strip()
            return self.generate_image(prompt)
        elif "describe image" in q_lower:
            image_path = query.replace("describe image", "", 1).strip()
            return self.handle_image_query(image_path)
        elif "weather" in q_lower:
            return self.access_real_time_data(query)
        else:
            return self.handle_text_query(query)
