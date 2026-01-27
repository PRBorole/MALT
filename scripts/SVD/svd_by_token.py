import torch
import argparse
import pandas as pd
import numpy as np
import torch._dynamo
import os
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import time 
import random
import re
import ast
from ast import literal_eval
from scipy import linalg
from tqdm import tqdm
from PIL import Image

import os
import sys

# Get the path of the Python script
current_dir = os.path.abspath(os.path.dirname(__file__))
# Exclude script name at the end
current_dir = os.path.split(current_dir)[0]+'/../'
print(current_dir)
sys.path.append(current_dir)

from src.utils import *

parser = argparse.ArgumentParser(description="usage help",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_name", help="path to config file")
parser.add_argument("--nsamples", help="samples for evaluation", type=int, default=1)
parser.add_argument("--ds_name", help="dataset to use")
parser.add_argument("--root_dir", help="root dir for project", default='.')
parser.add_argument("--attn_implementation", help="attention implementation", default='eager')

args = parser.parse_args()
model_name = args.model_name
ds_name = args.ds_name
nsamples = int(args.nsamples)
ROOT_DIR = args.root_dir  # ROOT_DIR for project
attn_implementation = args.attn_implementation

# model_name = 'molmoD-7B'
# ds_name = 'MathVision'
# nsamples = 50
# ROOT_DIR = '/exports/csce/eddie/inf/groups/ajitha_project/piyush/MALT/'
# attn_implementation = 'eager'

print(f"Model name: {model_name}, Dataset name: {ds_name}, nsamples: {nsamples}, ROOT_DIR: {ROOT_DIR}")
print(f"Model name: {type(model_name)}, Dataset name: {type(ds_name)}, nsamples: {type(nsamples)}, ROOT_DIR: {type(ROOT_DIR)}")

HF_dir = ROOT_DIR+'/models/'
if os.path.isdir(HF_dir):
    print(f"HF cache directory exists at: {HF_dir}")
else:
    raise FileNotFoundError(f"Directory does not exist: {HF_dir}")

os.environ["HF_HUB_CACHE"] = HF_dir # set your HF cache directory
os.environ["HF_HOME"] = HF_dir # set your HF cache directory

import sys
sys.path.append(ROOT_DIR+'/MALT/')
from src.utils import *
from src.probing import *
from src.utils_metrics import get_erank

random.seed(42)
results_path = f'{ROOT_DIR}/MALT/results/{model_name}/svd/'
prediction_df = pd.read_csv(f'{ROOT_DIR}/MALT/results/{model_name}/{model_name}_{ds_name}_prediction.csv')
prediction_corrected_df = pd.read_csv(f'{ROOT_DIR}/MALT/results/{model_name}/{model_name}_{ds_name}_prediction_corrected.csv')
image_dir_path = None

# load test dataset
if ds_name=='MATE':
    image_dir_path = f'{ROOT_DIR}/data/MATE-dev/img/'
    ds = load_jsonl_file(f'{ROOT_DIR}/data/MATE-dev/mm_0shot_llava_hfllava_1.5_7b_hf.jsonl')
    assert len(ds) == len(prediction_df), (
        f"Length mismatch: len(ds)={len(ds)} != len(prediction_df)={len(prediction_df)}"
    )
    ds = [ds[i] for i in prediction_corrected_df['idx_main'].to_list()]
    
else:
    ds = load_dataset_formatted(ds_name=ds_name)

    # filter to MCQ
    ds = ds.filter(lambda x: x["image"]!=None and (len(x["image"])==1 if type(x["image"])==list else 1))
    
    if len(ds)!=len(prediction_df):
        # filter to image input size <1900*1900 to avoid OOM
        ds = ds.filter(
            lambda x: x['image'].size[0]*x['image'].size[1]<1900*1900
            if type(x['image'])!=list 
            else x['image'][0].size[0]*x['image'][0].size[1]<1900*1900
        )
        
        if len(ds)!=len(prediction_df): # For some datasets, 1850*150 image size was used as limit as some datasets had HD images. 
            ds = ds.filter(
                lambda x: x['image'].size[0]*x['image'].size[1]<1850*1850
                if type(x['image'])!=list 
                else x['image'][0].size[0]*x['image'][0].size[1]<1850*1850
            )
            
            if len(ds) == len(prediction_df):
                pass
            else:
                ds = ds.filter(
                lambda x: x['image'].size[0]*x['image'].size[1]<1700*1700
                if type(x['image'])!=list 
                else x['image'][0].size[0]*x['image'][0].size[1]<1700*1700
            )
        
    assert len(ds) == len(prediction_df), (
        f"Length mismatch: len(ds)={len(ds)} != len(prediction_df)={len(prediction_df)}"
    )

    # filter to include only corrected idx
    ds = ds.select(prediction_corrected_df['idx_main'].to_list())

# subsample dataset
nsamples = min(nsamples, len(ds))
idx_ls = random.sample(range(len(ds)), nsamples)

if ds_name=='MATE':
    ds = [ds[i] for i in idx_ls] 
    prediction_corrected_df = prediction_corrected_df.iloc[idx_ls].reset_index()
else:
    ds = ds.select(idx_ls)
    prediction_corrected_df = prediction_corrected_df.iloc[idx_ls].reset_index()

    # Add prompt field
    ds = ds.map(
        lambda x: {"prompt": build_prompt(ds_item=x, ds_name=ds_name)},
        desc="Building prompts",
        batch_size=4
    )

# Load the model
model, processor, device = load_model_and_processor(
    model_name=model_name, 
    model_path=HF_dir, 
    attn_implementation=attn_implementation,
    torch_dtype=torch.float16 if 'mistral' in model_name else 'auto'
)

all_layer_embeddings_rank_ls = []
img_layer_embeddings_rank_ls = []
text_layer_embeddings_rank_ls = []
text_only_layer_embeddings_rank_ls = []
image_only_layer_embeddings_rank_ls = []

for idx in tqdm(range(len(ds)), desc='main'):
    batch = ( # We always use batch size 1 here
        ds[idx:idx+1] if isinstance(ds, list)
        else  ds.select(range(idx, idx+1))
    )

    images, prompts, inputs = process_input_data(
        ds=batch,  
        image_dir_path=image_dir_path,
        processor=processor,
        device=device,
        model_name=model_name
    )
    
    with torch.no_grad():
        # outputs = model(**{k:inputs[k][idx:idx+1] for k in inputs.keys()}, output_hidden_states=True)
        outputs = model(**inputs, output_hidden_states=True)
    model_embeddings = torch.vstack(outputs.hidden_states).cpu().detach().to(torch.float16).numpy()
    
    input_ids = inputs['input_ids'][0] # We are using batch size 1 always here
    if 'molmo' in model_name:
        image_tokens = (input_ids==processor.special_token_ids['<im_patch>']).cpu().detach().numpy().reshape(-1) 
    else:
        image_tokens = (input_ids==model.config.image_token_id).cpu().detach().numpy().reshape(-1)
    text_tokens = [False 
                   if i in list(processor.tokenizer.all_special_ids) else True 
                   for i in input_ids.detach().cpu().tolist()]
    # text_tokens = ~input_ids.detach().cpu().tolist().isin().cpu().detach().numpy().reshape(-1)
    
    all_layer_embeddings = model_embeddings.astype(np.float32)
    img_layer_embeddings = model_embeddings[:,image_tokens,:].astype(np.float32)
    text_layer_embeddings = model_embeddings[:,text_tokens,:].astype(np.float32)
    
    
    all_layer_embeddings_rank_ = []
    for i in tqdm(range(0,all_layer_embeddings.shape[0]), desc='all_layers'):
        all_layer_embeddings_rank_ = all_layer_embeddings_rank_ + [get_erank(all_layer_embeddings[i], device=device)]
    all_layer_embeddings_rank_ls = all_layer_embeddings_rank_ls + [all_layer_embeddings_rank_]
    print("all  done")
    
    img_layer_embeddings_rank_ = []
    for i in tqdm(range(0,img_layer_embeddings.shape[0]), desc='img_layers'):
        img_layer_embeddings_rank_ = img_layer_embeddings_rank_ + [get_erank(img_layer_embeddings[i], device=device)]
    img_layer_embeddings_rank_ls = img_layer_embeddings_rank_ls + [img_layer_embeddings_rank_]
    print("img  done")
    
    text_layer_embeddings_rank_ = []
    for i in tqdm(range(0,text_layer_embeddings.shape[0]), desc='text_layers'):
        text_layer_embeddings_rank_ = text_layer_embeddings_rank_ + [get_erank(text_layer_embeddings[i], device=device)]
    text_layer_embeddings_rank_ls = text_layer_embeddings_rank_ls + [text_layer_embeddings_rank_]
    print("text  done")
    
    ########### ONLY TEXT NO IMAGE TOKENS
    
    if 'molmo' in model_name:
        inputs_text_only = processor.process(images=None, text=prompts[0], padding=True, return_tensors="pt")
        inputs_text_only = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs_text_only.items()}
        
        text_only_tokens = [False 
                    if i in list(processor.tokenizer.all_special_ids) else True 
                    for i in inputs_text_only['input_ids'].detach().cpu().tolist()]
    else:
        inputs_text_only = processor(images=None, text=prompts, padding=True, return_tensors="pt")
        inputs_text_only = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs_text_only.items()}
        
        text_only_tokens = [False 
                    if i in list(processor.tokenizer.all_special_ids) else True 
                    for i in inputs_text_only['input_ids'][0].detach().cpu().tolist()]
    
    with torch.no_grad():
        # outputs = model(**{k:inputs[k][idx:idx+1] for k in inputs.keys()}, output_hidden_states=True)
        if len(inputs_text_only)==1:
            outputs = model(input_ids=inputs_text_only['input_ids'].reshape(1,-1), output_hidden_states=True)
        else:
            outputs = model(**inputs_text_only, output_hidden_states=True)
    text_only_layer_embeddings = torch.vstack(outputs.hidden_states).cpu().detach().to(torch.float16).numpy().astype(np.float32)
    text_only_layer_embeddings = text_only_layer_embeddings[:,text_only_tokens,:]
    
    
    text_only_layer_embeddings_rank_ = []
    for i in tqdm(range(0,text_only_layer_embeddings.shape[0]), desc='text_only_layers'):
        text_only_layer_embeddings_rank_ = text_only_layer_embeddings_rank_ + [get_erank(text_only_layer_embeddings[i], device=device)]
    text_only_layer_embeddings_rank_ls = text_only_layer_embeddings_rank_ls + [text_only_layer_embeddings_rank_] 
    print("text only done")

    
# all embedding df
all_ranks_df = pd.DataFrame(
    np.array(all_layer_embeddings_rank_ls),
    columns=range(0,all_layer_embeddings.shape[0])
)
all_ranks_df['Mean over'] = ['All tokens']*len(all_ranks_df)
all_ranks_df['idx_corrected'] = idx_ls   
all_ranks_df['idx_main'] = prediction_corrected_df['idx_main']
all_ranks_df['outputwImg'] = prediction_corrected_df['outputwImg']
all_ranks_df['predictionwImg'] = prediction_corrected_df['predictionwImg']
all_ranks_df['gold_referencewImg'] = prediction_corrected_df['gold_referencewImg'] 
all_ranks_df['tagwImg'] = prediction_corrected_df['tagwImg']
all_ranks_df['outputwoImg'] = prediction_corrected_df['outputwoImg']
all_ranks_df['predictionwoImg'] = prediction_corrected_df['predictionwoImg']
all_ranks_df['gold_referencewoImg'] = prediction_corrected_df['gold_referencewoImg']
all_ranks_df['tagwoImg'] = prediction_corrected_df['tagwoImg']


#img embedding df
img_ranks_df = pd.DataFrame(
    np.array(img_layer_embeddings_rank_ls),
    columns=range(0,img_layer_embeddings.shape[0])
)
img_ranks_df['Mean over'] = ['Img tokens']*len(img_ranks_df)
img_ranks_df['idx_corrected'] = idx_ls
img_ranks_df['idx_main'] = prediction_corrected_df['idx_main']
img_ranks_df['outputwImg'] = prediction_corrected_df['outputwImg']
img_ranks_df['predictionwImg'] = prediction_corrected_df['predictionwImg']
img_ranks_df['gold_referencewImg'] = prediction_corrected_df['gold_referencewImg'] 
img_ranks_df['tagwImg'] = prediction_corrected_df['tagwImg']
img_ranks_df['outputwoImg'] = prediction_corrected_df['outputwoImg']
img_ranks_df['predictionwoImg'] = prediction_corrected_df['predictionwoImg']
img_ranks_df['gold_referencewoImg'] = prediction_corrected_df['gold_referencewoImg']
img_ranks_df['tagwoImg'] = prediction_corrected_df['tagwoImg']

#text embedding df
text_ranks_df = pd.DataFrame(
    np.array(text_layer_embeddings_rank_ls),
    columns=range(0,text_layer_embeddings.shape[0])
)
text_ranks_df['Mean over'] = ['Text tokens']*len(text_ranks_df)
text_ranks_df['idx_corrected'] = idx_ls
text_ranks_df['idx_main'] = prediction_corrected_df['idx_main']
text_ranks_df['outputwImg'] = prediction_corrected_df['outputwImg']
text_ranks_df['predictionwImg'] = prediction_corrected_df['predictionwImg']
text_ranks_df['gold_referencewImg'] = prediction_corrected_df['gold_referencewImg'] 
text_ranks_df['tagwImg'] = prediction_corrected_df['tagwImg']
text_ranks_df['outputwoImg'] = prediction_corrected_df['outputwoImg']
text_ranks_df['predictionwoImg'] = prediction_corrected_df['predictionwoImg']
text_ranks_df['gold_referencewoImg'] = prediction_corrected_df['gold_referencewoImg']
text_ranks_df['tagwoImg'] = prediction_corrected_df['tagwoImg']

#text only embedding df
text_only_ranks_df = pd.DataFrame(
    np.array(text_only_layer_embeddings_rank_ls), 
    columns=range(0,text_only_layer_embeddings.shape[0])
)
text_only_ranks_df['Mean over'] = ['Text only tokens']*len(text_only_ranks_df)
text_only_ranks_df['idx_corrected'] = idx_ls
text_only_ranks_df['idx_main'] = prediction_corrected_df['idx_main']
text_only_ranks_df['outputwImg'] = prediction_corrected_df['outputwImg']
text_only_ranks_df['predictionwImg'] = prediction_corrected_df['predictionwImg']
text_only_ranks_df['gold_referencewImg'] = prediction_corrected_df['gold_referencewImg'] 
text_only_ranks_df['tagwImg'] = prediction_corrected_df['tagwImg']
text_only_ranks_df['outputwoImg'] = prediction_corrected_df['outputwoImg']
text_only_ranks_df['predictionwoImg'] = prediction_corrected_df['predictionwoImg']
text_only_ranks_df['gold_referencewoImg'] = prediction_corrected_df['gold_referencewoImg']
text_only_ranks_df['tagwoImg'] = prediction_corrected_df['tagwoImg']


result_df = pd.concat([all_ranks_df, img_ranks_df, text_ranks_df, text_only_ranks_df])
result_df.to_csv(f"{results_path}/{ds_name}_svd_effective_rank_corrected.csv")

