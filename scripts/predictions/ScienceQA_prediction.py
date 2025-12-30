import torch
import argparse
import pandas as pd
import torch._dynamo

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

from datasets import load_dataset

import sys
sys.path.append(current_dir)
from src.utils import *

# model_name = args.model_name
# 'molmoD-7B'
# 'llava-1.5-7b',
# 'llava-v1.6-mistral-7b', 
# 'llava-v1.6-vicuna-7b',
# 'Qwen2.5-VL-7B', 
# 'Qwen2.5-VL-3B', 
# 'Qwen3-VL-4B-Instruct', 
# 'Qwen3-VL-8B-Instruct', 

model_name = 'Qwen3-VL-8B-Instruct'
image_dir_path = None
attn_implementation = 'eager'
batch_size = 1
hint_use = False  # whether to include hint in the prompt
image_use = False # whether to include image in the prompt

experiment_tag = '' if hint_use else 'withoutHint'
experiment_tag = experiment_tag + ('' if image_use else 'withoutImg')


# load test dataset
ds = load_dataset("derek-thomas/ScienceQA", split="test")

# filter to MCQ
ds = ds.filter(lambda x: x["image"] is not None)

# Add image=None for every item
if not image_use:
    ds = ds.map(lambda x: {"image": None})

# Add prompt field
ds = ds.map(lambda ds_item: {**ds_item, "prompt": build_scienceqa_prompt(ds_item, hint_use=hint_use)})

if 'gemma' in model_name:
    torch._dynamo.config.cache_size_limit = 32

model, processor, device = load_model_and_processor(
    model_name=model_name, 
    model_path=HF_dir, 
    attn_implementation=attn_implementation,
    torch_dtype=torch.float16 if 'mistral' in model_name else 'auto'
)

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
    'gold_reference':  pd.DataFrame(ds)['answer'],
    'grade': [i['grade'] for i in ds]
})

print("match current vs gold reference: ",
     len(prediction_df[prediction_df['prediction']==prediction_df['gold_reference']]))

print("Accuracy: ",
     len(prediction_df[prediction_df['prediction']==prediction_df['gold_reference']])/len(prediction_df))

print("mismatch current vs gold reference: ",
     len(prediction_df[prediction_df['prediction']!=prediction_df['gold_reference']]))

prediction_df.to_csv(f"{current_dir}/results/{model_name}/{model_name}_{experiment_tag}_ScienceQA_prediction.csv")
