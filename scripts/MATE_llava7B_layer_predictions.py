from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import LlavaForConditionalGeneration, AutoModelForImageTextToText
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from PIL import Image
from src.utils import *
import pandas as pd
import numpy as np
from src.probing import *
import time 
import random
import re
import ast
import argparse
from scipy import linalg
from src.utils_metrics import *

random.seed(42)
start = time.time()

image_dir_path = './../datasets/MATE-dev/img/'
ds_main = load_jsonl_file('./../datasets/MATE-dev/mm_0shot_llava_hfllava_1.5_7b_hf.jsonl')
mate_df = pd.read_csv('./data/mate_df.csv',index_col=0)
linear_probe = True

attn_implementation = 'sdpa'
nsamples = 1
layers = 33

# Load the model
model, processor, device = load_model_and_processor(
    model_name='llava_1.5_7b', 
    attn_implementation=attn_implementation
)
model.eval()

metrics_dict = {'accuracy': [None] * layers}

if linear_probe:
    task_dict = {
        'color': ['gray', 'yellow', 'red', 'blue', 'green'],
        'material': ['rubber', 'metal'],
        'shape': ['cone', 'cylinder', 'cube'],
        'size':['0.35', '0.351', '0.7', '0.701']
    }

    prompt_type = 'complex' # 'simple', 'complex'
    mode = 'image' # 'image', 'text', 'image_and_text',
    nobjects = 3
    task = ['color', 'shape']
    target = ['gray', 'cylinder']
    results_path = f'./results/llava7B/linear_probing/layer_predictions'


    #  Create probe dataset 
    relevant_idx = mate_df[mate_df['object_count'] == nobjects]['idx'].to_list()
    pos_idx = mate_df[(mate_df['object_count'] == nobjects) & 
                    (mate_df[('_'.join(task))].apply(lambda x: '_'.join(target) in x))]['idx'].to_list()
    neg_idx = [i for i in relevant_idx if i not in pos_idx]

    neg_idx = random.sample(neg_idx, nsamples)
    pos_idx = random.sample(pos_idx, nsamples)

    probe_data_idx = pos_idx + neg_idx
    ds_main = [ds_main[idx].copy() for idx in probe_data_idx]
    gold_reference_binary = [1]*len(pos_idx) + [0]*len(neg_idx)

    # Set probe
    probe = LinearProbe(model=model,
                        nobjects=nobjects,
                        task=task,
                        mode=mode,
                        target=target,
                        prompt_type=prompt_type)


    for idx in range(len(ds_main)):
        ds_main[idx]['prompt'] = probe.get_prompt(ds_main[idx])

    metrics_dict = {
                    'accuracy': [None] * layers,
                    'f1_macro': [None] * layers,
                    'precision_macro': [None] * layers,
                    'recall_macro': [None] * layers,
                    'f1_micro': [None] * layers,
                    'precision_micro': [None] * layers,
                    'recall_micro': [None] * layers
    }


images, prompts, inputs = process_input_data(ds=ds_main,  
                                            image_dir_path=image_dir_path,
                                            processor=processor,
                                            device=device,
                                            model_name='llava_1.5_7b')


# Generate results for each layer
for layer in tqdm(range(layers), desc='layer'):
    y_pred = []
    
    for idx in tqdm(range(len(ds_main)), desc='probe_data'):
        decoder_input_ids = inputs["input_ids"][idx:idx+1]
        pixel_values = inputs["pixel_values"][idx:idx+1]
        attention_mask = inputs["attention_mask"][idx:idx+1]

        # We'll store generated tokens here
        generated = decoder_input_ids.clone()
        past_key_values = None
        # Generate exactly <> new tokens
        for step in range(10):
            with torch.no_grad():
                outputs = model(
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    input_ids=generated,       # feed only the last token
                    # past_key_values=past_key_values,   # speed via caching
                    # use_cache=True,                    # ensure caching is enabled
                    output_hidden_states=True,         # to get hidden states
                    return_dict=True
                )
            
            # outputs.decoder_hidden_states is a tuple: (embeddings, layer1, ..., last_layer)
            decoder_hidden_states = outputs.hidden_states

            # Extract the hidden state of the last layer, last token
            last_hidden = decoder_hidden_states[3][:, -1:, :]  # shape: (B, 1, hidden_size)

            # Convert to logits with the LM head
            with torch.no_grad():
                logits = model.lm_head(last_hidden)  # shape: (B, 1, vocab_size)

            # Choose next token (greedy here; you can replace with sampling logic)
            next_token_id = torch.argmax(logits, dim=-1)

            # Append token to the generated sequence
            generated = torch.cat([generated, next_token_id], dim=-1)
            attention_mask = torch.cat([attention_mask,torch.tensor([[1]],device=device)],dim=-1)


        # Decode the full generated sequence
        decoded = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
        # label = 'yes' if gold_reference_binary[idx] == 1 else 'no'

        y_pred = y_pred + [1 
                            if 'yes' in decoded.split('ASSISTANT:')[1].lower() 
                            else 0
                            ]
        
    # Metrics
    if linear_probe: 
        acc = accuracy_score(gold_reference_binary, y_pred)
        f1_macro = f1_score(gold_reference_binary, y_pred, average='macro')
        precsion_macro = precision_score(gold_reference_binary, y_pred, average='macro')
        recall_macro = recall_score(gold_reference_binary, y_pred, average='macro')

        f1_micro = f1_score(gold_reference_binary, y_pred, average='micro')
        precsion_micro = precision_score(gold_reference_binary, y_pred, average='micro')
        recall_micro = recall_score(gold_reference_binary, y_pred, average='micro')

        metrics_dict['accuracy'][layer] = acc
        metrics_dict['f1_macro'][layer] = f1_macro
        metrics_dict['precision_macro'][layer] = precsion_macro
        metrics_dict['recall_macro'][layer] = recall_macro
        metrics_dict['f1_micro'][layer] = f1_micro
        metrics_dict['precision_micro'][layer] = precsion_micro
        metrics_dict['recall_micro'][layer] = recall_micro
    else:
        acc = accuracy_score(gold_reference_binary, y_pred)
        metrics_dict['accuracy'][layer] = acc


print(get_gpu_memory())

all_model_outputs = batch_inference(
        ds_main, 
        image_dir_path, 
        model, processor, 
        device=device, 
        batch_size=8, 
        max_new_tokens=50, 
        model_name='llava_1.5_7b'
    )

if linear_probe:
    predictions = [i.split('ASSISTANT:')[1].lstrip() for i in all_model_outputs]

    y_pred = [1 if 'yes' in i.lower() else 0 for i in predictions]
    print(predictions)
    print(y_pred)
    print(get_gpu_memory())

    # Metrics
    acc = accuracy_score(gold_reference_binary, y_pred)

    f1_macro = f1_score(gold_reference_binary, y_pred, average='macro')
    precsion_macro = precision_score(gold_reference_binary, y_pred, average='macro')
    recall_macro = recall_score(gold_reference_binary, y_pred, average='macro')

    f1_micro = f1_score(gold_reference_binary, y_pred, average='micro')
    precsion_micro = precision_score(gold_reference_binary, y_pred, average='micro')
    recall_micro = recall_score(gold_reference_binary, y_pred, average='micro')

    metrics_dict['accuracy'] = metrics_dict['accuracy'] + [acc]
    metrics_dict['f1_macro'] = metrics_dict['f1_macro']+ [f1_macro]
    metrics_dict['precision_macro'] = metrics_dict['precision_macro'] + [precsion_macro]
    metrics_dict['recall_macro'] = metrics_dict['recall_macro'] + [recall_macro]
    metrics_dict['f1_micro'] = metrics_dict['f1_micro'] + [f1_micro]
    metrics_dict['precision_micro'] = metrics_dict['precision_micro'] + [precsion_micro]
    metrics_dict['recall_micro'] = metrics_dict['recall_micro'] + [recall_micro]

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df['layer'] = list(range(layers)) + ['final']

    # metrics_df.to_csv(f"{results_path}/layer_{mode}_{'_'.join(target)}_{nobjects}.csv")



