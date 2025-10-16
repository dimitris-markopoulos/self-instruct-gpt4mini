#==================================
# STEP 1 : INSTRUCTION GENERATION
#==================================
import random
import openai
import json
from pathlib import Path
import os
from dotenv import load_dotenv
import yaml
import re
from datetime import datetime
import sys
import time

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

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / f"bootstrap_{datetime.now():%Y%m%d_%H%M%S}.log"

def log(message: str, level: str = "INFO") -> None:
    """Prints a timestamped log message and writes to logs/bootstrap.log."""

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{now}] [{level.upper()}] {message}"
    print(formatted) # print to terminal
    with open(LOG_PATH, "a") as log_file: # also append to log file
        log_file.write(formatted + "\n")

log("=" * 60)
log(f"SELF-INSTRUCT STEP 1 | {datetime.now():%Y-%m-%d %H:%M:%S}")
log("=" * 60)

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
def generate_instructions(prompt: str, model="gpt-4o-mini") -> str:
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

##
def parse_instructions(text: str) -> list:
    """
    Parse raw model output (continuation of 'Task 9: ...') into a list of clean instruction dicts.
    """
    if not text.strip().startswith("Task"): # prepend "Task 9:" if model didn't include it
        text = "Task 9: " + text.strip()
    matches = re.findall(r"Task\s*\d+:\s*(.+?)(?=\s*Task\s*\d+:|$)", text, flags=re.S) # find all task blocks like "Task 9: some text"
    tasks = []
    for t in matches:
        cleaned = t.strip().replace("\n", " ").strip(" .")
        if cleaned:
            tasks.append({
                "instruction": cleaned,
                "source": "gpt-4o-mini",
            })
    return tasks

##
def create_task(model: str = "gpt-4o-mini", timeout: int = 45) -> list[dict]:
    """
    Wrapper of entire pipeline for step 1. 
    Outputs nice list[dict] with keys instruction and source.
    To use in loop to fill data/generated_task.jsonl
    """
    start = time.time()

    sample_list = grab_subsample()
    prompt = create_prompt(sample_list)
    try:
        while True:
            if time.time() - start > timeout:
                raise TimeoutError(f"API call exceeded {timeout}s, skipping batch.")
            generated_instructions_str = generate_instructions(prompt, model=model)
            break # if successful
    except Exception as e:
        log(f"(X) Skipping batch due to error: {repr(e)}", level="ERROR")
        return None

    task_list = parse_instructions(generated_instructions_str)
    duration = time.time() - start
    log(f"Batch finished in {duration:.1f}s — {len(task_list)} tasks.", level="INFO")
    return task_list

if __name__ == "__main__":
    start_time = datetime.now()
    n_iterations = 50
    log(f"Bootstrapping started for {n_iterations} iterations...")
    total_new = 0
    for i in range(1, n_iterations + 1):
        try:
            task_list = create_task()
            if not task_list: # if timeout occurs or api stalls
                log(f"Iteration {i}/{n_iterations}: skipped (no tasks returned).", level="WARN")
                continue
            with open("data/generated_tasks.jsonl", "a") as f:
                for task in task_list:
                    f.write(json.dumps(task) + "\n")
            total_new += len(task_list)
            log(f"Iteration {i}/{n_iterations}: {len(task_list)} new tasks generated.")
        except Exception as e:
            log(f"(X) Iteration {i}/{n_iterations} failed with error: {repr(e)}", level="ERROR")
        
        time.sleep(random.uniform(1.0, 2.0)) # to avoid rate limiting

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 60
    log(f"Bootstrapping complete. Total {total_new} new instructions appended "
        f"to generated_tasks.jsonl in {elapsed:.2f} minutes.")
