from openai import OpenAI

from app.config import OPENAI_API_KEY


class Generator:
    """Generate grounded answers using an OpenAI model."""

    def __init__(self, model="gpt-4o-mini"):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured.")

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    def generate(self, question, contexts):
        """Generate an answer from retrieved contexts."""

        from app.generation.prompt_builder import build_prompt

        prompt = build_prompt(question, contexts)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return {
            "answer": response.choices[0].message.content,
            "model": self.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
