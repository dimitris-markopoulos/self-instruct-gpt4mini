import os
from dotenv import load_dotenv
import openai
import yaml

load_dotenv("secrets.env")

with open("config.yaml") as f:
    raw_yaml = f.read()
    expanded_yaml = os.path.expandvars(raw_yaml) # Contains secret
    config = yaml.safe_load(expanded_yaml)

openai.api_key = config["openai"]["api_key"] # Configure the OpenAI client

# Simple test call
resp = openai.chat.completions.create(
    model=config["openai"]["model"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one short sentence."},
    ],
    temperature=config["openai"]["temperature"],
    max_tokens=config["openai"]["max_tokens"]
)

print("Response:", resp.choices[0].message.content)