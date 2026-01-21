import torch
import argparse
import pandas as pd
import torch._dynamo
impport random

import os
# # Get the path of the Python script
# current_dir = os.path.abspath(os.path.dirname(__file__))
# # Exclude script name at the end
# current_dir = os.path.split(current_dir)[0]+'/'
current_dir = '/exports/csce/eddie/inf/groups/ajitha_project/piyush/MALT/MALT/'
HF_dir = current_dir+'/../models/'
if os.path.isdir(HF_dir):
    print(f"HF cache directory exists at: {HF_dir}")
else:
    raise FileNotFoundError(f"Directory does not exist: {HF_dir}")

os.environ["HF_HUB_CACHE"] = HF_dir # set your HF cache directory
os.environ["HF_HOME"] = HF_dir # set your HF cache directory

import sys
sys.path.append(current_dir)
from src.utils import *
random.seed(42)

# model_name = args.model_name
# 'molmoD-7B'
# 'llava-1.5-7b',
# 'llava-v1.6-mistral-7b', 
# 'llava-v1.6-vicuna-7b',
# 'Qwen2.5-VL-7B', 
# 'Qwen2.5-VL-3B', 
# 'Qwen3-VL-4B-Instruct', 
# 'Qwen3-VL-8B-Instruct', 

# MATE,
# HallusionBench,
# MathVision,
# ScienceQA,
# MicroVQA,
# BLINK,
# VstarBench



model_name = 'Qwen2.5-VL-7B'
ds_name = 'VstarBench'
image_dir_path = None
attn_implementation = 'eager'
batch_size = 1
image_use = False  # whether to include image in the prompt
hint_use = False # Option only applicable for ScienceQA

experiment_tag = '' if hint_use or ds_name!='ScienceQA' else 'withoutHint'
experiment_tag = experiment_tag + ('' if image_use else 'withoutImg_')

# load test dataset
ds = load_dataset_formatted(ds_name=ds_name)

# filter to make sure only 1 image
ds = ds.filter(lambda x: x["image"]!=None and (len(x["image"])==1 if type(x["image"])==list else 1))

# filter to image input size <1700*1700 to avoid OOM
ds = ds.filter(
    lambda x: x['image'].size[0]*x['image'].size[1]<1900*1900 # 1700 for qwen2.5 7B Vstar
    if type(x['image'])!=list 
    else x['image'][0].size[0]*x['image'][0].size[1]<1900*1900
)

# Add image=None for every item
if not image_use:
    ds = ds.map(lambda x: {"image": None})

# Add prompt field
ds = ds.map(
    lambda x: {"prompt": build_prompt(ds_item=x, ds_name=ds_name)},
    desc="Building prompts",
    batch_size=64
)

if 'gemma' in model_name:
    torch._dynamo.config.cache_size_limit = 32

# model, processor, device = load_model_and_processor(
#     model_name=model_name, 
#     model_path=HF_dir, 
#     attn_implementation=attn_implementation,
#     torch_dtype=torch.float16 if 'mistral' in model_name else 'auto'
# )

all_model_outputs = batch_inference(
    ds=ds, 
    image_dir_path=image_dir_path, 
    image_use=image_use,
    model=model, 
    processor=processor, 
    device=device, 
    batch_size=batch_size, 
    max_new_tokens=50, 
    model_name=model_name
)

predictions = process_predictions(model_name=model_name, all_model_outputs=all_model_outputs)

prediction_df = pd.DataFrame({
    'output': all_model_outputs,
    'prediction': [
        int(i) 
        if i.isdigit() 
        else (
            int(i[0]) 
            if len(i) > 0 and i[0].isdigit() 
            else 10000 #random large number for non-digit answers
        ) for i in predictions
    ],
    'gold_reference':  ds[:]['gold_reference'],
    'tag':  ds[:]['tag']
})

print("mismatch current vs gold reference: ",
     len(prediction_df[prediction_df['prediction']!=prediction_df['gold_reference']]))

print("match current vs gold reference: ",
     len(prediction_df[prediction_df['prediction']==prediction_df['gold_reference']]))

print("Accuracy: ",
     len(prediction_df[prediction_df['prediction']==prediction_df['gold_reference']])/len(prediction_df))

prediction_df.to_csv(f"{current_dir}/results/{model_name}/{model_name}_{experiment_tag}{ds_name}_prediction.csv")


