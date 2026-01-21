import os
from openai import OpenAI


def generate_risk_commentary(payload: dict, prompt_template: str) -> str:
    """
    Generate LLM-based risk commentary.
    Initialized lazily to avoid API dependency when agent is disabled.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None:
        raise RuntimeError("OPENAI_API_KEY not set but agent was enabled.")

    client = OpenAI(api_key=api_key)

    prompt = prompt_template.format(**payload)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a senior quantitative risk analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
