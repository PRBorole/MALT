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
nsamples = 5

# Load the model
model, processor, device = load_model_and_processor(
    model_name='llava_1.5_7b', 
    attn_implementation=attn_implementation
)
model.eval()

task_dict = {
    'color': ['gray', 'yellow', 'red', 'blue', 'green'],
    'material': ['rubber', 'metal'],
    'shape': ['cone', 'cylinder', 'cube'],
    'size':['0.35', '0.351', '0.7', '0.701']
}

prompt_type = 'complex' # 'simple', 'complex'
mode = 'image_and_text' # 'image', 'text', 'image_and_text',
nobjects = 3
task = ['color']
target = ['gray']
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


images, prompts, inputs = process_input_data(ds=ds_main,  
                                            image_dir_path=image_dir_path,
                                            processor=processor,
                                            device=device,
                                            model_name='llava_1.5_7b')

print(get_gpu_memory())

all_model_outputs = []
for idx in tqdm(range(len(prompts))):
    inputs_text_only = processor(images=images[idx], text=prompts[idx], padding=True, return_tensors="pt")
    input_ids = inputs_text_only['input_ids'].to(device)
    attention_mask = inputs_text_only['attention_mask'].to(device)
    # input_ids = torch.cat([inputs_text_only['input_ids'][:,0:1], inputs_text_only['input_ids'][:,2:]], dim=1).to(device)
    # attention_mask = torch.cat([inputs_text_only['attention_mask'][:,0:1], inputs_text_only['attention_mask'][:,2:]], dim=1).to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=50)
        model_answers = processor.batch_decode(outputs, skip_special_tokens=True)
        all_model_outputs = all_model_outputs + model_answers
        outputs = outputs.detach().cpu()
    



predictions = [i.split('ASSISTANT:')[1].lstrip() for i in all_model_outputs]
print(predictions)

y_pred = [1 if 'yes' in i.lower() else 0 for i in predictions]

print(y_pred)
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

# # Metrics
# acc = accuracy_score(gold_reference_binary, y_pred)

# f1_macro = f1_score(gold_reference_binary, y_pred, average='macro')
# precsion_macro = precision_score(gold_reference_binary, y_pred, average='macro')
# recall_macro = recall_score(gold_reference_binary, y_pred, average='macro')

# f1_micro = f1_score(gold_reference_binary, y_pred, average='micro')
# precsion_micro = precision_score(gold_reference_binary, y_pred, average='micro')
# recall_micro = recall_score(gold_reference_binary, y_pred, average='micro')

# metrics_dict['accuracy'] = metrics_dict['accuracy'] + [acc]
# metrics_dict['f1_macro'] = metrics_dict['f1_macro']+ [f1_macro]
# metrics_dict['precision_macro'] = metrics_dict['precision_macro'] + [precsion_macro]
# metrics_dict['recall_macro'] = metrics_dict['recall_macro'] + [recall_macro]
# metrics_dict['f1_micro'] = metrics_dict['f1_micro'] + [f1_micro]
# metrics_dict['precision_micro'] = metrics_dict['precision_micro'] + [precsion_micro]
# metrics_dict['recall_micro'] = metrics_dict['recall_micro'] + [recall_micro]

# metrics_df = pd.DataFrame(metrics_dict)
# metrics_df['layer'] = list(range(layers)) + ['final']

# metrics_df.to_csv(f"{results_path}/layer_{mode}_{'_'.join(target)}_{nobjects}.csv")



