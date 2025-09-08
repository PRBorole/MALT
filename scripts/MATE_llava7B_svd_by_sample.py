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
from src.utils_metrics import get_erank

random.seed(42)
start = time.time()

image_dir_path = './../datasets/MATE-dev/img/'
ds_main = load_jsonl_file('./../datasets/MATE-dev/mm_0shot_llava_hfllava_1.5_7b_hf.jsonl')
mate_df = pd.read_csv('./data/mate_df.csv',index_col=0)

task_dict = {
    'color': ['gray', 'yellow', 'red', 'blue', 'green'],
    'material': ['rubber', 'metal'],
    'shape': ['cone', 'cylinder', 'cube'],
    'size':['0.35', '0.351', '0.7', '0.701']
}

prompt_type = 'complex' # 'simple', 'complex'
mode = 'image' # 'image', 'text', 'image_and_text',
nobjects = 10
task = ['color','shape', 'material']
target = ['gray', 'cylinder', 'metal']
# nsamples = 50
mean_dim = 1 # 1 for Sentence pool 2 for token pool of embeddings

results_path = f'./results/llava7B/svd/sample/'

attn_implementation = 'sdpa'

# Create probe dataset 
relevant_idx = mate_df[mate_df['object_count'] == nobjects]['idx'].to_list()
pos_idx = mate_df[(mate_df['object_count'] == nobjects) & 
                  (mate_df[('_'.join(task))].apply(lambda x: '_'.join(target) in x))]['idx'].to_list()
neg_idx = [i for i in relevant_idx if i not in pos_idx]

# neg_idx = random.sample(neg_idx, nsamples)
# pos_idx = random.sample(pos_idx, nsamples)

probe_data_idx = pos_idx + neg_idx
probe_data = [ds_main[idx].copy() for idx in probe_data_idx]
gold_reference_binary = [1]*len(pos_idx) + [0]*len(neg_idx)

# Load the model
model, processor, device = load_model_and_processor(
    model_name='llava_1.5_7b', 
    attn_implementation=attn_implementation
)


# Set probe
probe = LinearProbe(model=model,
                    nobjects=nobjects,
                    task=task,
                    mode=mode,
                    target=target,
                    prompt_type=prompt_type)


for idx in range(len(probe_data)):
    probe_data[idx]['prompt'] = probe.get_prompt(probe_data[idx])


_, _, inputs = process_input_data(ds=probe_data,  
                                    image_dir_path=image_dir_path,
                                    processor=processor,
                                    device=device,
                                    model_name='llava_1.5_7b')


# embeddings
all_layer_embeddings = [None]*len(probe_data)
img_layer_embeddings = [None]*len(probe_data)
text_layer_embeddings = [None]*len(probe_data)

for idx in tqdm(range(len(probe_data))):
    model_embeddings = probe.get_layer_llava_embeddings(pixel_values=inputs['pixel_values'][idx:idx+1],
                                                        input_ids=inputs['input_ids'][idx:idx+1],
                                                        attention_mask=inputs['attention_mask'][idx:idx+1],
                                                        mean_dim=mean_dim,   # For both level embeddings, 1 for sentence level, 2 for token level
                                                        special_token_embeddings=False) 
    
    all_layer_embeddings[idx] = model_embeddings['all_layer_embeddings']
    img_layer_embeddings[idx] = model_embeddings['img_layer_embeddings']
    text_layer_embeddings[idx] = model_embeddings['text_layer_embeddings']

all_layer_embeddings = np.array(all_layer_embeddings).astype(np.float32)
img_layer_embeddings = np.array(img_layer_embeddings).astype(np.float32)
text_layer_embeddings = np.array(text_layer_embeddings).astype(np.float32)

# Ranks
all_layer_embeddings_rank_ls = []
img_layer_embeddings_rank_ls = []
text_layer_embeddings_rank_ls = []
text_only_layer_embeddings_rank_ls = []

# for i in tqdm(range(0,all_layer_embeddings.shape[1]), desc='all_layer_embeddings'):
#     all_layer_embeddings_rank_ls = all_layer_embeddings_rank_ls + [np.linalg.matrix_rank(all_layer_embeddings[:,i,:])]
# print(f"all_layer_embeddings_rank_ls: {all_layer_embeddings_rank_ls}")

# for i in tqdm(range(0,img_layer_embeddings.shape[1]), desc='img_layer_embeddings'):
#     img_layer_embeddings_rank_ls = img_layer_embeddings_rank_ls + [np.linalg.matrix_rank(img_layer_embeddings[:,i,:])]
# print(f"img_layer_embeddings_rank_ls: {img_layer_embeddings_rank_ls}")

# for i in tqdm(range(0,text_layer_embeddings.shape[1]), desc='text_layer_embeddings'):
#     text_layer_embeddings_rank_ls = text_layer_embeddings_rank_ls + [np.linalg.matrix_rank(text_layer_embeddings[:,i,:])]
# print(f"text_layer_embeddings_rank_ls: {text_layer_embeddings_rank_ls}")

for i in tqdm(range(0,all_layer_embeddings.shape[1]), desc='all_layer_embeddings'):
    all_layer_embeddings_rank_ls = all_layer_embeddings_rank_ls + [get_erank(all_layer_embeddings[:,i,:], device=device)]
print(f"all_layer_embeddings_rank_ls: {all_layer_embeddings_rank_ls}")

for i in tqdm(range(0,img_layer_embeddings.shape[1]), desc='img_layer_embeddings'):
    img_layer_embeddings_rank_ls = img_layer_embeddings_rank_ls + [get_erank(img_layer_embeddings[:,i,:], device=device)]
print(f"img_layer_embeddings_rank_ls: {img_layer_embeddings_rank_ls}")

for i in tqdm(range(0,text_layer_embeddings.shape[1]), desc='text_layer_embeddings'):
    text_layer_embeddings_rank_ls = text_layer_embeddings_rank_ls + [get_erank(text_layer_embeddings[:,i,:], device=device)]
print(f"text_layer_embeddings_rank_ls: {text_layer_embeddings_rank_ls}")

########### ONLY TEXT NO IMAGE TOKENS
text_only_layer_embeddings = [None]*len(probe_data)
for idx in tqdm(range(len(probe_data))):
    inputs_text_only = processor(images=None, text=probe_data[idx]['prompt'], padding=True, return_tensors="pt")
    inputs_text_only = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs_text_only.items()}

    input_ids = inputs_text_only['input_ids']
    attention_mask = inputs_text_only['attention_mask']

    with torch.no_grad():
        inputs_embeds = model.get_input_embeddings()(input_ids)

    # Forward pass with output_hidden_states=True
    with torch.no_grad():
        outputs = model.language_model(attention_mask=attention_mask,
                                        images=None,
                                        inputs_embeds=inputs_embeds,
                                        output_hidden_states=True,
                                        output_attentions=False,
                                        padding=True,
                                        return_dict=True)

    text_only_layer_embeddings[idx] =  torch.vstack(outputs.hidden_states).mean(axis=mean_dim).detach().cpu().numpy().astype(np.float32)

text_only_layer_embeddings = np.array(text_only_layer_embeddings)
text_only_layer_embeddings.shape

# for i in tqdm(range(0,text_only_layer_embeddings.shape[1]),desc='text only'):
#     text_only_layer_embeddings_rank_ls = text_only_layer_embeddings_rank_ls + [np.linalg.matrix_rank(text_only_layer_embeddings[:,i,:])]
# print(f"idx: {idx}, ranks: {text_only_layer_embeddings_rank_ls}")

for i in tqdm(range(0,text_only_layer_embeddings.shape[1]),desc='text only'):
    text_only_layer_embeddings_rank_ls = text_only_layer_embeddings_rank_ls + [get_erank(text_only_layer_embeddings[:,i,:], device=device)]
print(f"idx: {idx}, ranks: {text_only_layer_embeddings_rank_ls}")

# store results
all_ranks_df = pd.DataFrame(
    np.array(all_layer_embeddings_rank_ls),
)
all_ranks_df['Mean over'] = ['All']*len(all_ranks_df)
all_ranks_df['prompt'] = ['With Image']*len(all_ranks_df)
all_ranks_df['layers'] = list(range(len(all_ranks_df)))


img_ranks_df = pd.DataFrame(
    np.array(img_layer_embeddings_rank_ls)
)
img_ranks_df['Mean over'] = ['Image']*len(img_ranks_df)
img_ranks_df['prompt'] = ['With Image']*len(img_ranks_df)
img_ranks_df['layers'] = list(range(len(img_ranks_df)))

text_ranks_df = pd.DataFrame(
    np.array(text_layer_embeddings_rank_ls)
)
text_ranks_df['Mean over'] = ['Text']*len(text_ranks_df)
text_ranks_df['prompt'] = ['With Image']*len(text_ranks_df)
text_ranks_df['layers'] = list(range(len(text_ranks_df)))

text_only_ranks_df = pd.DataFrame(
    np.array(text_only_layer_embeddings_rank_ls)
)
text_only_ranks_df['Mean over'] = ['Text only']*len(text_only_ranks_df)
text_only_ranks_df['prompt'] = ['Without Image']*len(text_only_ranks_df)
text_only_ranks_df['layers'] = list(range(len(text_only_ranks_df)))

result_df = pd.concat([all_ranks_df, img_ranks_df, text_ranks_df, text_only_ranks_df])
# result_df
result_df.to_csv(f"{results_path}/svd_effective_rank_{mode}_{'_'.join(task)}_{'_'.join(target)}_{nobjects}.csv")





