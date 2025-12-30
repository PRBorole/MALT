import torch
import argparse
import pandas as pd
import torch._dynamo
import os

# # Get the path of the Python script
# current_dir = os.path.abspath(os.path.dirname(__file__))
# # Exclude script name at the end
# current_dir = os.path.split(current_dir)[0]+'/'
# print(current_dir)

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

# parser = argparse.ArgumentParser(description="usage help",
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--model_name", help="path to config file")

args = parser.parse_args()
model_name = args.model_name

# 'molmoD-7B'
# 'llava-1.5-7b',
# 'llava-v1.6-mistral-7b', 
# 'llava-v1.6-vicuna-7b',
# 'Qwen2.5-VL-7B', 
# 'Qwen2.5-VL-3B', 
# 'Qwen3-VL-4B-Instruct', 
# 'Qwen3-VL-8B-Instruct', 

model_name = 'Qwen3-VL-8B-Instruct'
ds_name = 'MATE'
image_dir_path = None
attn_implementation = 'eager'
batch_size = 1
image_use = False # whether to include image in the prompt
hint_use = False # Option only applicable for ScienceQA

experiment_tag = '' if hint_use or ds_name!='ScienceQA' else 'withoutHint'
experiment_tag = experiment_tag + ('' if image_use else 'withoutImg_')

print(f"experiment_tag: {experiment_tag}")

if model_name=='gemma3-4B' or model_name=='gemma3-12B':
    torch._dynamo.config.cache_size_limit = 32

attn_implementation = 'eager'
model_path = current_dir+'/../models/'

image_dir_path = f'{current_dir}/../data/MATE-dev/img/'

ds = load_jsonl_file(f'{current_dir}/../data/MATE-dev/mm_0shot_llava_hfllava_1.5_7b_hf.jsonl')

if not image_use:
    print("Not using image")
    image_dir_path = None
    for idx, _ in enumerate(ds):
        ds[idx]['image'] = None

# mate_df = pd.read_csv(f'{current_dir}/data/mate_df.csv',index_col=0)

model, processor, device = load_model_and_processor(
    model_name=model_name, 
    model_path=model_path, 
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


predictions = process_predictions(model_name, all_model_outputs)


prediction_df = pd.DataFrame({
    'image':[ds[i]['image'] for i in range(len(ds))],
    'output':all_model_outputs,
    'prediction': predictions,
    'gold_reference': pd.DataFrame(ds)['gold_reference'],
    'object_count': [data['object_count'] for data in ds]
})


prediction_df['prediction_lower'] = prediction_df['prediction'].apply(lambda x: x.lower())
prediction_df['gold_reference_lower'] = prediction_df['gold_reference'].apply(lambda x: x.lower())
prediction_df['gold_reference_formatted'] = prediction_df['gold_reference_lower'].apply(lambda x: x.split(':')[1].lstrip().replace('"','').replace('}',''))

print("mismatch current vs gold reference: ",
    len(prediction_df[prediction_df['prediction_lower']!=prediction_df['gold_reference_formatted']]))

# prediction_df[prediction_df['prediction_lower']!=prediction_df['gold_reference_formatted']][['prediction_lower','gold_reference_formatted']]

if model_name == 'llava-1.5-7b':
    original_pred = [i['prediction'] for i in ds]
    prediction_df['paper_prediction'] = original_pred
    prediction_df['paper_prediction_lower'] = prediction_df['paper_prediction'].apply(lambda x: x.lower())
    
    print("mismatch current vs MATE paper prediction: ",
        len(prediction_df[prediction_df['prediction_lower']!=prediction_df['paper_prediction_lower']]))
    
    print("mismatch MATE paper prediction vs gold reference: ",
        len(prediction_df[prediction_df['paper_prediction_lower']!=prediction_df['gold_reference_lower']]))

prediction_df.to_csv(f'{current_dir}/results/{model_name}/{model_name}_{experiment_tag}MATE_prediction.csv')

