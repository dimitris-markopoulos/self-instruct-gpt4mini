import openai
import json
from tqdm import tqdm
import yaml
from pathlib import Path
import os
from dotenv import load_dotenv
import re
import time
import random

# Load API KEY
load_dotenv("secrets.env") # store secret
with open("config.yaml") as f:
    raw_yaml = f.read()
    expanded_yaml = os.path.expandvars(raw_yaml) # Contains secret
    config = yaml.safe_load(expanded_yaml)
openai.api_key = config["openai"]["api_key"] # Configure the OpenAI client


with open("data/generated_tasks.jsonl", "r") as f:
    instructions = [json.loads(line) for line in f]

def create_classification_prompt(input_task : str) -> bool:
    return f"""
    Can the following task be regarded as a classification task with finite output labels?

    Task: Given my personality and the job, tell me if I would be suitable.
    Is it classification? Yes

    Task: Give me an example of a time when you had to use your sense of humor.
    Is it classification? No

    Task: Replace the placeholders in the given text with appropriate named entities.
    Is it classification? No

    Task: Fact checking - tell me if the statement is true, false, or unknown, based on your
    knowledge and common sense.
    Is it classification? Yes

    Task: Return the SSN number for the person.
    Is it classification? No

    Task: Detect if the Reddit thread contains hate speech.
    Is it classification? Yes

    Task: Analyze the sentences below to identify biases.
    Is it classification? No

    Task: Select the longest sentence in terms of the number of words in the paragraph, output
    the sentence index.
    Is it classification? Yes

    Task: Find out the toxic word or phrase in the sentence.
    Is it classification? No

    Task: Rank these countries by their population.
    Is it classification? No

    Task: You are provided with a news article, and you need to identify all the categories that
    this article belongs to. Possible categories include: Music, Sports, Politics, Tech, Finance,
    Basketball, Soccer, Tennis, Entertainment, Digital Game, World News. Output its categories one
    by one, seperated by comma.
    Is it classification? Yes

    Task: Given the name of an exercise, explain how to do it.
    Is it classification? No

    Task: Select the oldest person from the list.
    Is it classification? Yes

    Task: Find the four smallest perfect numbers.
    Is it classification? No

    Task: Does the information in the document supports the claim? You can answer "Support" or
    "Unsupport".
    Is it classification? Yes

    Task: Create a detailed budget for the given hypothetical trip.
    Is it classification? No

    Task: Given a sentence, detect if there is any potential stereotype in it. If so, you should
    explain the stereotype. Else, output no.
    Is it classification? No

    ⋯

    Task: To make the pairs have the same analogy, write the fourth word.
    Is it classification? No

    Task: Given a set of numbers, find all possible subsets that sum to a given number.
    Is it classification? No

    Task: {input_task}
    """

def generate_bool(prompt: str) -> str:
    """
    Query OpenAI api to classify task instructions.
    """
    resp = openai.chat.completions.create(
        model=config["openai"]["model"],
        messages=[ 
            {"role": "system", "content": (
                "Continue the classification of tasks below. "
                "Output only 'Yes' or 'No'."
                "Do not explain, comment, or ask for clarification."
            )},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=64
    )
    return resp.choices[0].message.content

def parse_for_bool(response_text: str) -> bool | None:
    """
    Use regular expression to pull bool from 'Is it classification? bool'
    """

    text = response_text.strip().lower()
    match = re.search(r'\b(yes|no)\b', text)
    if not match:
        return None
    return match.group(1) == "yes"

def run_classification_pipeline(task: str) -> tuple[str, bool]:
    """
    Simple wrapper to run classification pipeline.
    """
    prompt = create_classification_prompt(task)
    response_text = generate_bool(prompt)
    bool_val = parse_for_bool(response_text)
    return (task, bool_val)
    
if __name__ == "__main__":
    for item in tqdm(instructions, desc="Classifying tasks"):
        task = item["instruction"]
        try:
            task, bool_val = run_classification_pipeline(task)
        
        except Exception as e:
            continue
    
        with open("data/classified_tasks.jsonl", "a") as f:
            json.dump({"instruction": task, "is_classification": bool_val}, f)
            f.write("\n")
        
        time.sleep(random.uniform(0.5, 1.0)) # to avoid rate limiting