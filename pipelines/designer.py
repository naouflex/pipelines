"""A manifold to integrate OpenAI's ImageGen models into Open-WebUI"""

import os
from typing import List, Union, Generator, Iterator

from pydantic import BaseModel

from openai import OpenAI

class Pipeline:
    """OpenAI ImageGen pipeline"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""

        OPENAI_API_BASE_URL: str = "https://api.openai.com/v1"
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
        IMAGE_SIZE: str = "1792x1024"
        NUM_IMAGES: int = 1
        PROMPT_MODEL: str = "gpt-4-turbo"  # Model to use for prompt generation

    def __init__(self):
        self.type = "manifold"
        self.name = "Inverse Designer: "

        self.valves = self.Valves()
        self.client = OpenAI(
            base_url=self.valves.OPENAI_API_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
        )

        self.pipelines = self.get_openai_assistants()

    async def on_startup(self) -> None:
        """This function is called when the server is started."""
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        """This function is called when the server is stopped."""
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        """This function is called when the valves are updated."""
        print(f"on_valves_updated:{__name__}")
        self.client = OpenAI(
            base_url=self.valves.OPENAI_API_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
        )
        self.pipelines = self.get_openai_assistants()

    def get_openai_assistants(self) -> List[dict]:
        """Get the available ImageGen models from OpenAI

        Returns:
            List[dict]: The list of ImageGen models
        """

        if self.valves.OPENAI_API_KEY:
            models = self.client.models.list()
            return [
                {
                    "id": model.id,
                    "name": model.id,
                }
                for model in models
                if "dall-e" in model.id
            ]

        return []

    def generate_optimized_prompt(self, context: str, user_message: str) -> str:
        """Generate an optimized image prompt using an LLM"""
        try:
            system_prompt = """You are an expert at writing prompts for DALL-E image generation. 
            Convert the user's request and conversation context into a detailed, vivid prompt that will produce the best possible image.
            Focus on visual details, style, lighting, and composition. Return only the prompt text, without any explanations."""

            response = self.client.chat.completions.create(
                model=self.valves.PROMPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\nUser Request: {user_message}"}
                ],
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating optimized prompt: {e}")
            return user_message  # Fallback to original message if prompt generation fails

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        try:
            # Build context from previous messages
            context = ""
            for message in messages:
                if isinstance(message.get("content"), str):
                    role = message.get("role", "")
                    content = message.get("content", "")
                    context += f"{role}: {content}\n"

            # Generate optimized prompt using LLM
            optimized_prompt = self.generate_optimized_prompt(context, user_message)
            print(f"Optimized prompt: {optimized_prompt}")  # For debugging

            message = ""
            for _ in range(self.valves.NUM_IMAGES):
                response = self.client.images.generate(
                    model=model_id,
                    prompt=optimized_prompt,
                    size=self.valves.IMAGE_SIZE,
                    n=1,
                )

                for image in response.data:
                    if image.url:
                        message += "![image](" + image.url + ")\n"

            yield message
        except Exception as e:
            print(f"Error generating image: {e}")
            yield f"Error generating image: {e}"
