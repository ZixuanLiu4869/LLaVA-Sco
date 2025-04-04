# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.


import copy
import itertools
import torch
import os
import json
from PIL import Image
from datasets import Dataset

# check system prompt token seq or user prompt token seq is in the current token list
def check_header(targets,seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] in targets:
            return True
    return False
def replace_target(target,seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] == target:
            seq[i],seq[i+1],seq[i+2] = -100,-100,-100
    return seq
def tokenize_dialogs(dialogs, images, processor):
    text_prompt = processor.apply_chat_template(dialogs)
    batch = processor(images=images, text=text_prompt,padding = True, return_tensors="pt")
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i,n in enumerate(labels) if n == 128009]
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        prompt_header_seqs = [[128006, 9125, 128007],[128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx:idx+1]
            if check_header(prompt_header_seqs,current_seq):
                # found prompt header, indicating that this seq should be masked
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
            else:
                last_idx = idx+1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq,labels)
        # Mask the padding token and image token 128256 
        for i in range(len(labels)):
            if labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256: #  128256 is image token index
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch


def get_custom_dataset(dataset_config, processor, split, split_ratio=1):

    data_path = "llava-sco-100k.jsonl"
    image_base_path = "YOUR/PATH/TO/IMAGE"
    
    
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    samples = []
    for entry in data:
        if "image" not in entry:
            continue
        image_file = os.path.join(image_base_path, entry["image"])
        #image = Image.open(image_file)
        image = image_file
        
        texts = []
        conversation_pair = {}
        for conversation in entry["conversations"]:
            if conversation["from"] == "human":
                conversation_pair["user"] = conversation["value"]
            elif conversation["from"] == "gpt":
                conversation_pair["assistant"] = conversation["value"]
            
            if "user" in conversation_pair and "assistant" in conversation_pair:
                texts.append(conversation_pair)
                conversation_pair = {} 
                
        sample = {
            "images": [image],  
            "texts": texts     
        }
        samples.append(sample)

    dataset = Dataset.from_dict({
        "images": [sample["images"] for sample in samples],
        "texts": [sample["texts"] for sample in samples]
    })
    return dataset

class CoTDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
    def __call__(self, samples):
        dialogs,images = [],[]
        for sample in samples:
            image_list,sample_list = sample["images"],sample["texts"]
            if len(image_list) > 1:
                raise ValueError("Only support one image per sample")
            #image = Image.open(image_list[0]['path']).convert("RGB") 
            image = Image.open(image_list[0]).convert("RGB")
            dialog = []
            for sample_dict in sample_list:
                if not dialog:
                    dialog += [
                    {"role":"user","content":[{"type": "image"},{"type": "text", "text": sample_dict["user"].strip()}]},
                    {"role":"assistant","content":[{"type": "text", "text": sample_dict["assistant"].strip()}]}
                ]
                
                else:
                    dialog += [
                    {"role":"user","content":[{"type": "text", "text": sample_dict["user"].strip()}]},
                    {"role":"assistant","content":[{"type": "text", "text": sample_dict["assistant"].strip()}]}
                ]
            dialogs.append(dialog)
            images.append([image])
        return tokenize_dialogs(dialogs,images, self.processor)
def get_data_collator(processor):
    return CoTDataCollator(processor)
