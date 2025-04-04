import os
from openai import AzureOpenAI
import base64
from datasets import load_dataset
from tqdm import tqdm
import json 
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from itertools import islice
import numpy as np

tqdm.pandas()


# Set up Azure OpenAI credentials
AZURE_OPENAI_API_KEY = "YOUR/OPENAI/KEY"
AZURE_OPENAI_ENDPOINT = "YOUR/ENDPOINT"

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-08-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Checkpoint file
CHECKPOINT_FILE = "llava_cot_checkpoint.json"
FINAL_OUTPUT_FILE = "llava-cot-100k-refined2.json"
SAVE_INTERVAL = 100  # Save every 1000 processed samples
MAX_WORKERS = 8  # Number of parallel processes (adjust based on your hardware)

# Function to encode image files to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def think_twice(entry):
    """
    Uses GPT-4o to refine an initial answer by thinking twice while strictly maintaining the original format.
    If the initial answer is incorrect, it will be corrected while keeping the exact structure.
    
    Args:
        question (str): The original question.
        initial_answer (str): The initial answer generated by GPT, following the required format.
        
    Returns:
        dict: A dictionary containing the original and refined answers, both strictly following the given format.
    """

    question = entry["question"]
    image_path = entry["image_path"]  # Path to the image file
    initial_answer = entry["initial_answer"]

    # Encode image
    base64_image = encode_image(image_path)

    # Step 1: Ask GPT-4o to reflect and refine while preserving structure
    system_prompt = """
    You are an AI assistant that critically evaluates and improves responses while strictly maintaining their original structure.
    You MUST follow the exact format provided: <SUMMARY>, <CAPTION>, <REASONING>, and <CONCLUSION>.
    - Preserve all section headers.
    - Do NOT change the formatting or remove any sections.
    - The initial answer is correct, however you should reevalute the answer again and improve the SUMMARY, CAPTION and REASONING phase, especialy the REASONING phase.
    - Ensure that the final answer inside CONCLUSION is correct.
    """

    # User prompt
    user_prompt = f"""
    Here is a question and its initial answer along with an image. Let's reevaluate this again and improve the response if possible, but strictly keep the same format.

    **Question:** {question}

    **Initial Answer:**
    {initial_answer}

    Now, provide a refined response while strictly following the format:
    <SUMMARY> ... </SUMMARY>
    <CAPTION> ... </CAPTION>
    <REASONING> ... </REASONING>
    <CONCLUSION> ... </CONCLUSION>
    """

    retries = 5
    wait_time = 60
    while retries > 0:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", 
                     "content": [
                         {"type": "text", "text": user_prompt},
                         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},          
                     ]}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            
            refined_answer = response.choices[0].message.content
            return {
                "question": question,
                "original_answer": initial_answer,
                "refined_answer": refined_answer,
                "image_path": image_path
            }
        
        except Exception as e:
            print(f"API Error: {e}. Retrying ({6 - retries}/5)...")
            retries -= 1
            time.sleep(wait_time)  # Wait before retrying

    print(f"Skipping question due to repeated failures: {question}")
    return {
        "question": question,
        "original_answer": initial_answer,
        "refined_answer": "",  # Mark failed attempts
        "image_path": image_path
    }

train_dataset = load_dataset("Xkev/LLaVA-CoT-100k", split="train")

original_columns = train_dataset.column_names

def process(samples):
    image = ["image/"+img for img in samples["image"]]
    conversations = samples["conversations"]
    prompt = [ c[0]["value"] for c in conversations]
    answer_from_gpt = [c[1]["value"] for c in conversations]

    return{
        "question": prompt,
        "image_path": image, 
        "initial_answer": answer_from_gpt,
    }

num_proc = 24
train_dataset = train_dataset.map(process, batched=True, num_proc=num_proc,remove_columns=original_columns)

# Load checkpoint if exists
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        refined_dataset = json.load(f)
    print(f"Resuming from checkpoint: {len(refined_dataset)} samples processed.")
else:
    refined_dataset = []


# Resume processing from last checkpoint
start_index = len(refined_dataset)
remaining_data = train_dataset[start_index:]  # Get only unprocessed data

# Function to process dataset in parallel
def process_batch(entry):
    return [think_twice(entry)]

start = time.time()

# Parallel processing
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = []
    
    for index in range(start_index, len(train_dataset)):
        future = executor.submit(process_batch, train_dataset[index])
        futures.append(future)
    
    for future in as_completed(futures):
        try:
            results = future.result()
            refined_dataset.extend(results)
            print(len(refined_dataset))
            # Save checkpoint every SAVE_INTERVAL samples
            if len(refined_dataset) % SAVE_INTERVAL == 0:
                with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                    json.dump(refined_dataset, f, indent=4)
                print(f"Checkpoint saved at {len(refined_dataset)} samples.")
                end = time.time()
                print(f"Time for processing 100 data is: {end-start}")
                start = time.time()

        except Exception as e:
            print(f"Error processing batch: {e}")

# Final save
with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(refined_dataset, f, indent=4)

print("Processing complete. Final results saved.")