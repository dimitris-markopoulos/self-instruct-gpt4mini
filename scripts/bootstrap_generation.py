import random
import openai
import json
from pathlib import Path
import os
from dotenv import load_dotenv
import yaml

# Load API KEY
load_dotenv("secrets.env") # store secret
with open("config.yaml") as f:
    raw_yaml = f.read()
    expanded_yaml = os.path.expandvars(raw_yaml) # Contains secret
    config = yaml.safe_load(expanded_yaml)
openai.api_key = config["openai"]["api_key"] # Configure the OpenAI client

# Download the 175 human-written tasks
with open("data/seed_tasks.jsonl") as f:
    seed_tasks = [json.loads(line) for line in f]

##
def grab_subsample():
    """
    Pulls 6 human-written instructions and 2 LLM instructions. 
    Initalizes with 8 human-written instructions.
    """
    with open("data/generated_tasks.jsonl") as f:
        sampled_llm_tasks = [json.loads(line) for line in f]
    
    if not sampled_llm_tasks: 
        # intialize to all human-written instructions if no generated responses exist
        return random.sample(seed_tasks, 8)
    
    llm_sample = random.sample(sampled_llm_tasks, 2)
    human_sample = random.sample(seed_tasks, 6)

    return llm_sample + human_sample

##
def create_prompt(instruction_sample : list) -> str:
    """
    Render the instruction-generation prompt (using Table 5 from SELF-INSTRUCT).
    """
    header = "Come up with a series of tasks:\n\n"
    lines = []
    for i, task in enumerate(instruction_sample):
        line = f'Task {i+1}: {task["instruction"].strip(" ")}'
        lines.append(line)
        if i == 7: # add last section (empty); for LLM to fill in
            lines.append(f'Task {i+2}:')
    return "\n".join(lines)

##
def generate_instructions(prompt: str, model="gpt-4o-mini"):
    """
    Query OpenAI api to generate new task instructions.
    """
    resp = openai.chat.completions.create(
        model=config["openai"]["model"],
        messages=[ 
            # had to include role because model was being too helpful and did not continue list
            # this gave it further direction - original research did not do this
            {"role": "system", "content": (
                "Continue the numbered list of tasks below. "
                "Write 3 to 8 new Task entries that follow the same style. "
                "Do not explain, comment, or ask for clarification."
            )},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        top_p=0.5,
        max_tokens=1024
    )

    return resp.choices[0].message.content

if __name__ == '__main__':
    sample_list = grab_subsample()
    prompt = create_prompt(sample_list)
    # generate_instructions(prompt, model="gpt-4o-mini")

