# llm_helper.py
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_disease_info(disease_name: str) -> str:
    """
    Generate information and cure for a given plant disease using OpenAI GPT.
    """
    prompt = (
        f"Provide a short description and simple cure or treatment for the plant disease: {disease_name}."
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating disease info: {e}"
