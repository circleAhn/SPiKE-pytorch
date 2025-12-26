# mainly modified from: https://github.com/ziqiangcui/CoLaKG-SIGIR25/blob/main/llm_code/llm_request_api.py

import pandas as pd
import numpy as np
import gc
import pickle
import random
import torch
import copy
import asyncio
import aiohttp
import json
import time
import re
import aiofiles
import os
from openai import OpenAI

data_type = "user" # item or user
data_name = 'book' # choose a dataset

input_path = f"datasets/kg/amazon-{data_name}_kg/{data_type}_raw_profiles_colakg2.json"    
write_path = f"datasets/kg/amazon-{data_name}_kg/{data_type}_llm_profiles_colakg2.json" 

if data_type == "item":
    if data_name == "book":
        system_input = """Assume you are an expert in book recommendation.
You are given information about a book, including its title, category, related descriptions, and a few user reviews.
Please help (i) fill in any missing background knowledge, (ii) summarize the book’s core theme or message, and (iii) analyze what kind of readers would be most interested in this book by drawing on relevant evidence or insights to support your analysis.
Your response should be a coherent paragraph and should not exceed 150 words."""
    elif data_name == "amazon_movie":
        system_input = """Assume you are an expert in movie and TV recommendation.
You are given information about a title, including its name, genre, synopsis, production details, and selected viewer reviews.
Please help (i) fill in any missing background knowledge, (ii) summarize the core theme or message of the title, and (iii) analyze what kind of viewers would be most interested in this movie or show by drawing on relevant evidence or insights to support your analysis.
Your response should be a coherent paragraph and should not exceed 150 words."""
    elif data_name == "yelp":
        system_input = """Assume you are an expert in restaurant and business recommendation.
You are given information about a business, including its name, category, location, and a few customer reviews.
Please help (i) provide any relevant background knowledge that may be missing, (ii) summarize the business’s key offerings or atmosphere, and (iii) analyze what types of customers would be most interested in this place, based on available descriptions and reviews.
Your response should be a coherent paragraph and should not exceed 150 words."""
elif data_type == "user":
    if data_name == "book":
        system_input = """Assume you are an expert in book recommendation with access to a user's book-reading history.
You are given user information that includes descriptions of books the user has interacted with, including each book’s title, category, related information, and the user’s own reviews.
Please analyze and summarize this user's reading preferences in terms of literary tastes, topical interests, and stylistic preferences.
Your response should be a coherent and fluent paragraph, not exceeding 100 words."""
    elif data_name == "amazon_movie":
        system_input = """Assume you are an expert in movie and TV recommendation with access to a user's viewing history.
You are given user information that includes descriptions of titles the user has watched, including each title’s name, genre, content summary, and the user’s own reviews.
Please analyze and summarize this user's viewing preferences in terms of genre tendencies, thematic interests, and stylistic preferences.
Your response should be a coherent and fluent paragraph, not exceeding 100 words."""
    elif data_name == "yelp":
        system_input = """Assume you are an expert in restaurant and business recommendation with access to a user's review history.
You are given user information that includes descriptions of businesses they have visited, including each business’s name, category, location, and the user’s own reviews.
Please analyze and summarize the user’s preferences in terms of food types, service expectations, and ambiance preferences.
Your response should be a coherent and fluent paragraph, not exceeding 100 words."""
elif data_type == "entity":
    if data_name == "book":
        system_input = """Assume you are an expert in book recommendation.
You are given the name of an entity, along with descriptions of books associated with it. 
Please help (i) provide any missing background knowledge about the entity, (ii) summarize the entity’s key role or relevance based on the associated books, and (iii) analyze what kind of readers would be most interested in this entity’s works by drawing on relevant evidence or patterns observed in the associated books.
Your response should be a coherent paragraph and should not exceed 150 words."""
    elif data_name == "amazon_movie":
        system_input = """Assume you are an expert in movie and TV recommendation.
You are given the name of an entity (such as a director, actor, composer, or production label), along with descriptions of titles associated with it.
Please help (i) provide any missing background knowledge about the entity, (ii) summarize the entity’s key role or contribution based on the associated titles, and (iii) analyze what kind of viewers would be most interested in this entity’s works by drawing on relevant evidence or patterns observed in the associated titles.
Your response should be a coherent paragraph and should not exceed 150 words."""
    elif data_name == "yelp":
        system_input = """Assume you are an expert in restaurant and business recommendation.
You are given the name of an entity (e.g., a city, state, or business category), along with descriptions of businesses associated with it.
Please help (i) provide any relevant background knowledge about the entity, (ii) summarize the entity’s significance or characteristics based on the associated businesses, and (iii) analyze what types of customers are likely to be drawn to this entity, based on observed patterns.
Your response should be a coherent paragraph and should not exceed 150 words."""

# Load the dictionary from the file
with open(input_path, 'r', encoding="utf-8") as f:
    input_dic = json.load(f)

api_key = "" # input your own api_key
base_url = "https://api.openai.com/v1/chat/completions"

openai = OpenAI(api_key = api_key)

requests = [
    {"role": "user", "content": request, "item_key":key}
    for key, request in list(input_dic.items())#[:256]
]


concurrent_limit = 50 
semaphore = asyncio.Semaphore(concurrent_limit)

batch_size = 512
max_retries = 10
batch_delay = 5

async def fetch_response(session, request, retries=0):
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini-2024-07-18",
            "messages": [
                {"role": "system", "content": system_input},
                {"role": "user", "content": request["content"]},
                #request,
            ],
            "temperature": 0.0,
            "top_p": 0.001,
            "stream": False
        }
        try:
            async with session.post(base_url, headers=headers, data=json.dumps(payload), timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                else:
                    print(f"Error: Received status code {response.status}")
                    return None
        except (aiohttp.ClientConnectorError, aiohttp.ClientConnectorSSLError, asyncio.TimeoutError) as e:
            if retries < max_retries:
                print(f"Retrying... ({retries + 1}/{max_retries}) due to {e}")
                return await fetch_response(session, request, retries + 1)
            else:
                print(f"Failed after {max_retries} retries due to {e}.")
                return None

async def process_batch(session, batch):
    tasks = [fetch_response(session, request) for request in batch]
    return await asyncio.gather(*tasks)

async def write_responses_to_file(responses, file_path):
    async with aiofiles.open(file_path, 'w') as f:
        await f.write(json.dumps(responses, indent=4))

async def main(file_path):
    start_time = time.time()  
    responses = {} 

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=concurrent_limit)) as session:
        for i in range(0, len(requests), batch_size):
            print(f"Processing batch {i // batch_size + 1}")
            batch = requests[i:i + batch_size]
            batch_responses = await process_batch(session, batch)
            for request, response in zip(batch, batch_responses):
                item_key = request['item_key']
                responses[item_key] = response

            await write_responses_to_file(responses, file_path)
            print(f"Batch {i // batch_size + 1} completed, sleeping for {batch_delay} seconds")
            await asyncio.sleep(batch_delay)

    end_time = time.time()  
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    return responses

if os.path.exists(write_path):
    os.remove(write_path)
    print(f"Deleted existing file: {write_path}")

responses = asyncio.run(main(write_path))
print(".......", len(list(responses.keys())), ".......")
