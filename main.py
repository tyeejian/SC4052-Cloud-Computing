from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import requests
from dotenv import load_dotenv
import os
import base64

print("Initializing....")
load_dotenv()

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# messages = [
#     {"role": "system", 
#     "content": """
#     You are a professional documentation writer. You can write in-depth documentation or README based on code. 
    
#     Your task is to refurbished exisiting README when given one, write README of a given code, or write a summary of a given code. Follow the rules below.

#     1. Skip reading the line "Search: " in the first line.
#     2. A new README should include minimally
#         - Note saying this is generated from LLM (Italics)
#         - Table of contents
#         - About
#         - How to build
#         - Documentation
#         - License
#         - Contacts
#         - Technology stack used (if applicable)

#     Include in your own knowledge what is needed for a README. Use the context below

#     {context}

#     """},
# ]



messages = [
    {"role": "system", 
    "content": """
    You are a helpful AI Assitant. Your task is to create a summary descriptions of a code function provided by the user.

    In the summary description, explain as simple as possible, as if explaining to a beginner in coding.
    """},
]

# Apply the chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize and move to model device
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Create a streamer to print the output as it's generated
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

print("Done initialized!")

# # Stream generation
# _ = model.generate(
#     **model_inputs,
#     max_new_tokens=512,
#     streamer=streamer,
#     pad_token_id=tokenizer.eos_token_id
# )

def reply(prompt):
    messages.append({"role" : "user", "content" : prompt})
    # Apply the chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize and move to model device
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print("Qwen: ", end="\n")
    # Stream generation
    _ = model.generate(
        **model_inputs,
        max_new_tokens=512,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id
    )


def search_code(params):
    url = 'https://api.github.com/search/code'
    headers = {
        "Accept": "application/vnd.github.v3+json", 
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization" : f"Bearer {os.environ.get('GITHUB_TOKEN')}"
    }
    params = {
        "q" : "addClass+in:file+language:js+repo:jquery/jquery"
    }

    # Send the GET request
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        items = data.get("items", [])
        for item in items:
            print(f"Name: {item['name']}")
            print(f"Path: {item['path']}")
            print(f"Repository: {item['repository']['full_name']}")
            print(f"URL: {item['html_url']}")
            print("-" * 40)
        item = items[0]
        git_url = item['git_url']
        response = requests.get(git_url)
        if response.status_code == 200:
            git_data = response.json()
            git_content = git_data.get('content')
            content = base64.b64decode(git_content).decode('utf-8')
            print(content)
        reply(content)
        
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    while True:
        print("1. Search Github User")
        print("2. Search Github Code")
        prompt = input("Type your message(q to quit): ")
        if (prompt == "q"):
            break
        elif (prompt == "1"):
            print(1)
        elif (prompt == "2"):
            search_code("Test")
            # reply(prompt=prompt)
