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
from jive.AJIVE import AJIVE

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
mode = 'image_and_text' # 'image', 'text', 'image_and_text',
nobjects = 7
task = ['color', 'shape']
target = ['gray','cylinder']
nsamples = 1

results_path = f'./results/llava7B/svd/ajive/'

attn_implementation = 'sdpa'

# Create probe dataset 
relevant_idx = mate_df[mate_df['object_count'] == nobjects]['idx'].to_list()
pos_idx = mate_df[(mate_df['object_count'] == nobjects) & 
                  (mate_df[('_'.join(task))].apply(lambda x: '_'.join(target) in x))]['idx'].to_list()
neg_idx = [i for i in relevant_idx if i not in pos_idx]

neg_idx = random.sample(neg_idx, nsamples)
pos_idx = random.sample(pos_idx, nsamples)

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


images, prompts, inputs = process_input_data(ds=probe_data,  
                                            image_dir_path=image_dir_path,
                                            processor=processor,
                                            device=device,
                                            model_name='llava_1.5_7b')


all_layer_embeddings_rank_ls = []
img_layer_embeddings_rank_ls = []
text_layer_embeddings_rank_ls = []
text_only_layer_embeddings_rank_ls = []
image_only_layer_embeddings_rank_ls = []
common_layer_embeddings_rank_ls = []
image_individual_layer_embeddings_rank_ls = []
text_individual_layer_embeddings_rank_ls = []

for idx in tqdm(range(len(probe_data)), desc='probe_data'):
    model_embeddings = probe.get_layer_llava_embeddings(pixel_values=inputs['pixel_values'][idx:idx+1],
                                                        input_ids=inputs['input_ids'][idx:idx+1],
                                                        attention_mask=inputs['attention_mask'][idx:idx+1],
                                                        mean_dim='None',   # For both level embeddings, 1 for sentence level, 2 for token level
                                                        special_token_embeddings=False) 
    
    all_layer_embeddings = model_embeddings['all_layer_embeddings']
    img_layer_embeddings = model_embeddings['img_layer_embeddings']
    text_layer_embeddings = model_embeddings['text_layer_embeddings']

    all_layer_embeddings = np.array(all_layer_embeddings).astype(np.float32)
    img_layer_embeddings = np.array(img_layer_embeddings).astype(np.float32)
    text_layer_embeddings = np.array(text_layer_embeddings).astype(np.float32)


    all_layer_embeddings_rank_ = []
    for i in range(0,all_layer_embeddings.shape[0]):
        all_layer_embeddings_rank_ = all_layer_embeddings_rank_ + [get_erank(all_layer_embeddings[i], device=device)]
    all_layer_embeddings_rank_ls = all_layer_embeddings_rank_ls + [all_layer_embeddings_rank_]

    img_layer_embeddings_rank_ = []
    for i in range(0,img_layer_embeddings.shape[0]):
        img_layer_embeddings_rank_ = img_layer_embeddings_rank_ + [get_erank(img_layer_embeddings[i], device=device)]
    img_layer_embeddings_rank_ls = img_layer_embeddings_rank_ls + [img_layer_embeddings_rank_]

    text_layer_embeddings_rank_ = []
    for i in range(0,text_layer_embeddings.shape[0]):
        text_layer_embeddings_rank_ = text_layer_embeddings_rank_ + [get_erank(text_layer_embeddings[i], device=device)]
    text_layer_embeddings_rank_ls = text_layer_embeddings_rank_ls + [text_layer_embeddings_rank_]

    ########### ONLY TEXT NO IMAGE TOKENS
    inputs_text_only = processor(images=None, text=prompts[idx], padding=True, return_tensors="pt")
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

    text_only_layer_embeddings =  torch.vstack(outputs.hidden_states).detach().cpu().numpy().astype(np.float32)
    text_only_layer_embeddings_rank_ = []

    for i in range(0,text_only_layer_embeddings.shape[0]):
        text_only_layer_embeddings_rank_ = text_only_layer_embeddings_rank_ + [get_erank(text_only_layer_embeddings[i], device=device)]
    text_only_layer_embeddings_rank_ls = text_only_layer_embeddings_rank_ls + [text_only_layer_embeddings_rank_] 

    ########### ONLY IMAGE NO TEXT TOKENS
    inputs_image_only = processor(images=images[idx], text='<image>', padding=True, return_tensors="pt")
    inputs_image_only = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs_image_only.items()}

    pixel_values = inputs_image_only['pixel_values']
    input_ids = inputs_image_only['input_ids']
    attention_mask = inputs_image_only['attention_mask']

    vision_feature_layer = None
    vision_feature_select_strategy = None
    vision_feature_layer = (vision_feature_layer if vision_feature_layer is not None else model.config.vision_feature_layer)
    vision_feature_select_strategy = (vision_feature_select_strategy
                                        if vision_feature_select_strategy is not None
                                        else model.config.vision_feature_select_strategy)

    with torch.no_grad():
        inputs_embeds = model.get_input_embeddings()(input_ids)

    with torch.no_grad():
        image_features = model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy
            )
        image_features = torch.cat(image_features, dim=0)

    special_image_mask = input_ids == model.config.image_token_id
    n_image_tokens = (special_image_mask).sum()
    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    if inputs_embeds[special_image_mask].numel() != image_features.numel():
        n_image_features = image_features.shape[0] * image_features.shape[1]
        raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    # Forward pass with output_hidden_states=True
    with torch.no_grad():
        outputs = model.language_model(attention_mask=attention_mask,
                                        images=images[idx],
                                        inputs_embeds=inputs_embeds,
                                        output_hidden_states=True,
                                        output_attentions=False,
                                        padding=True,
                                        return_dict=True)

    image_only_layer_embeddings =  torch.vstack(outputs.hidden_states).detach().cpu().numpy().astype(np.float32)
    image_only_layer_embeddings_rank_ = []
    
    for i in range(0,image_only_layer_embeddings.shape[0]):
        image_only_layer_embeddings_rank_ = image_only_layer_embeddings_rank_ + [get_erank(image_only_layer_embeddings[i], device=device)]
    image_only_layer_embeddings_rank_ls = image_only_layer_embeddings_rank_ls + [image_only_layer_embeddings_rank_] 

    # We have now computed all the svds, time to run AJIVE
    common_layer_embeddings_rank_ = []
    image_individual_layer_embeddings_rank_ = []
    text_individual_layer_embeddings_rank_ = []
    for i in tqdm(range(0,all_layer_embeddings.shape[0]), desc='ajive'):
        blocks = {'image': img_layer_embeddings[i].T, 'text': text_layer_embeddings[i].T}
        init_signal_ranks = {'image': int(img_layer_embeddings_rank_[i]), 'text': int(text_layer_embeddings_rank_[i])}

        erank = get_ajive_erank(blocks, init_signal_ranks, device=device)
        common_layer_embeddings_rank_ = common_layer_embeddings_rank_ + [erank['common']]
        image_individual_layer_embeddings_rank_ = image_individual_layer_embeddings_rank_ + [erank['image']]
        text_individual_layer_embeddings_rank_ = text_individual_layer_embeddings_rank_ + [erank['text']]

    common_layer_embeddings_rank_ls = common_layer_embeddings_rank_ls + [common_layer_embeddings_rank_] 
    image_individual_layer_embeddings_rank_ls = image_individual_layer_embeddings_rank_ls + [image_individual_layer_embeddings_rank_] 
    text_individual_layer_embeddings_rank_ls = text_individual_layer_embeddings_rank_ls + [text_individual_layer_embeddings_rank_] 


### SAVE dfs
# all embedding df
all_ranks_df = pd.DataFrame(
    np.array(all_layer_embeddings_rank_ls),
    columns=range(0,all_layer_embeddings.shape[0])
)

all_ranks_df['contains'] = ['both img and text']*len(all_ranks_df)
all_ranks_df['label'] = gold_reference_binary
all_ranks_df['sample'] = list(range(len(all_ranks_df)))
all_ranks_df['Mean over'] = ['All tokens']*len(all_ranks_df)

#img embedding df
img_ranks_df = pd.DataFrame(
    np.array(img_layer_embeddings_rank_ls),
    columns=range(0,img_layer_embeddings.shape[0])
)

img_ranks_df['contains'] = ['both img and text']*len(img_ranks_df)
img_ranks_df['label'] = gold_reference_binary
img_ranks_df['sample'] = list(range(len(img_ranks_df)))
img_ranks_df['Mean over'] = ['Img tokens']*len(img_ranks_df)

#text embedding df
text_ranks_df = pd.DataFrame(
    np.array(text_layer_embeddings_rank_ls),
    columns=range(0,text_layer_embeddings.shape[0])
)
text_ranks_df['contains'] = ['both img and text']*len(text_ranks_df)
text_ranks_df['label'] = gold_reference_binary
text_ranks_df['sample'] = list(range(len(text_ranks_df)))
text_ranks_df['Mean over'] = ['Text tokens']*len(text_ranks_df)

#text only embedding df
text_only_ranks_df = pd.DataFrame(
    np.array(text_only_layer_embeddings_rank_ls), 
    columns=range(0,text_only_layer_embeddings.shape[0])
)
text_only_ranks_df['contains'] = ['only text']*len(text_only_ranks_df)
text_only_ranks_df['label'] = gold_reference_binary
text_only_ranks_df['sample'] = list(range(len(text_only_ranks_df)))
text_only_ranks_df['Mean over'] = ['Text only tokens']*len(text_only_ranks_df)

#image only embedding df
image_only_ranks_df = pd.DataFrame(
    np.array(image_only_layer_embeddings_rank_ls), 
    columns=range(0,image_only_layer_embeddings.shape[0])
)
image_only_ranks_df['contains'] = ['only img']*len(image_only_ranks_df)
image_only_ranks_df['label'] = gold_reference_binary
image_only_ranks_df['sample'] = list(range(len(image_only_ranks_df)))
image_only_ranks_df['Mean over'] = ['Image only tokens']*len(image_only_ranks_df)

#ajive common rank df
common_ranks_df = pd.DataFrame(
    np.array(common_layer_embeddings_rank_ls), 
    columns=range(0,all_layer_embeddings.shape[0])
)
common_ranks_df['contains'] = ['both img and text']*len(common_ranks_df)
common_ranks_df['label'] = gold_reference_binary
common_ranks_df['sample'] = list(range(len(common_ranks_df)))
common_ranks_df['Mean over'] = ['ajive common rank']*len(common_ranks_df)

#ajive text rank df
ajive_text_ranks_df = pd.DataFrame(
    np.array(text_individual_layer_embeddings_rank_ls), 
    columns=range(0,all_layer_embeddings.shape[0])
)
ajive_text_ranks_df['contains'] = ['both img and text']*len(ajive_text_ranks_df)
ajive_text_ranks_df['label'] = gold_reference_binary
ajive_text_ranks_df['sample'] = list(range(len(ajive_text_ranks_df)))
ajive_text_ranks_df['Mean over'] = ['ajive text rank']*len(ajive_text_ranks_df)


# ajive image rank df
ajive_image_ranks_df = pd.DataFrame(
    np.array(image_individual_layer_embeddings_rank_ls), 
    columns=range(0,all_layer_embeddings.shape[0])
)
ajive_image_ranks_df['contains'] = ['both img and text']*len(ajive_image_ranks_df)
ajive_image_ranks_df['label'] = gold_reference_binary
ajive_image_ranks_df['sample'] = list(range(len(ajive_image_ranks_df)))
ajive_image_ranks_df['Mean over'] = ['ajive image rank']*len(ajive_image_ranks_df)

result_df = pd.concat([
    all_ranks_df, 
    img_ranks_df, 
    text_ranks_df, 
    text_only_ranks_df, 
    image_only_ranks_df,
    common_ranks_df,
    ajive_image_ranks_df,
    ajive_text_ranks_df
])
result_df.to_csv(f"{results_path}/erank_{mode}_{'_'.join(target)}_{nobjects}.csv")
