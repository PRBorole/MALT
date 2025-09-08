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
import itertools

random.seed(42)
start = time.time()


image_dir_path = './../datasets/MATE-dev/img/'
ds_main = load_jsonl_file('./../datasets/MATE-dev/mm_0shot_llava_hfllava_1.5_7b_hf.jsonl')
mate_df = pd.read_csv('./data/mate_df.csv',index_col=0)


task_dict = {
    'color': ['gray', 'yellow', 'red', 'blue', 'green', 'brown', 'purple', 'cyan'],
    'material': ['rubber', 'metal'],
    'shape': ['cone', 'cylinder', 'cube', 'sphere'],
    'size':['0.35', '0.351', '0.7', '0.701']
}

prompt_type = 'simple' # 'simple', 'complex'
mode = 'image' # 'image', 'text', 'image_and_text', 'count'
results_path = f'./results/llava7B/linear_probing/probe_{mode}/mutate/'
batch_inference_mode = 1 # get predicitons
attn_implementation = 'sdpa'

nobjects = 7
task = ['color', 'shape']
old_target = ['gray', 'cylinder']

# Create probe dataset 
relevant_idx = mate_df[mate_df['object_count'] == nobjects]['idx'].to_list()
pos_idx = mate_df[(mate_df['object_count'] == nobjects) & 
                  (mate_df[('_'.join(task))].apply(lambda x: '_'.join(old_target) in x))]['idx'].to_list()

if len(pos_idx)>len(relevant_idx)/2:
    neg_idx = [i for i in relevant_idx if i not in pos_idx]
    pos_idx = random.sample(pos_idx, len(neg_idx))
else:
    neg_idx = random.sample([i for i in relevant_idx if i not in pos_idx], len(pos_idx))

probe_data_idx = pos_idx + neg_idx
probe_data = [ds_main[idx].copy() for idx in probe_data_idx]

count_dict = {
   '_'.join(i):0 for i in itertools.product(*[task_dict[t] for t in task])
}

for idx in range(len(probe_data)):
    target_ls = ['_'.join([ob[t] for t in task]) for ob in probe_data[idx]['scene']['objects']]
    for target in target_ls:
        count_dict[target] = count_dict[target] + 1

### auto select new target, highest occuring 
new_target = ''
new_target_count = 0
for k,v in count_dict.items():
    if k!='_'.join(old_target):
        if v>new_target_count:
            new_target = k
            new_target_count = v


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
                    target=old_target,
                    prompt_type=prompt_type)

gold_reference_binary = []
for idx in range(len(probe_data)):
    probe_data[idx]['prompt'] = probe.get_prompt(probe_data[idx])
    if new_target in ['_'.join([ob[t] for t in task]) for ob in probe_data[idx]['scene']['objects']]:
        gold_reference_binary = gold_reference_binary + [1]
    else:
        gold_reference_binary = gold_reference_binary + [0]



_, _, inputs = process_input_data(ds=probe_data,  
                                    image_dir_path=image_dir_path,
                                    processor=processor,
                                    device=device,
                                    model_name='llava_1.5_7b')



##### sentence & token level embedding
s_all_layer_embeddings = [None]*len(probe_data)
s_img_layer_embeddings = [None]*len(probe_data)
s_text_layer_embeddings = [None]*len(probe_data)
# s_special_layer_embeddings = [None]*len(probe_data)
t_all_layer_embeddings = [None]*len(probe_data)
t_img_layer_embeddings = [None]*len(probe_data)
t_text_layer_embeddings = [None]*len(probe_data)

for idx in tqdm(range(len(probe_data))):
    model_embeddings = probe.get_layer_llava_embeddings(pixel_values=inputs['pixel_values'][idx:idx+1],
                                                        input_ids=inputs['input_ids'][idx:idx+1],
                                                        attention_mask=inputs['attention_mask'][idx:idx+1],
                                                        mean_dim='both',   # For both level embeddings, 1 for sentence level, 2 for token level
                                                        special_token_embeddings=False) 
    
    s_all_layer_embeddings[idx] = model_embeddings['ax1_all_layer_embeddings']
    t_all_layer_embeddings[idx] = model_embeddings['ax2_all_layer_embeddings']
    s_img_layer_embeddings[idx] = model_embeddings['ax1_img_layer_embeddings']
    t_img_layer_embeddings[idx] = model_embeddings['ax2_img_layer_embeddings']
    s_text_layer_embeddings[idx] = model_embeddings['ax1_text_layer_embeddings']
    t_text_layer_embeddings[idx] = model_embeddings['ax2_text_layer_embeddings']

s_all_layer_embeddings = np.array(s_all_layer_embeddings) 
t_all_layer_embeddings = np.array(t_all_layer_embeddings)
s_img_layer_embeddings = np.array(s_img_layer_embeddings)
t_img_layer_embeddings = np.array(t_img_layer_embeddings)
s_text_layer_embeddings = np.array(s_text_layer_embeddings)
t_text_layer_embeddings = np.array(t_text_layer_embeddings)


# Probing experiment sentence all
s_all = probe.probing_experiment(layer_embeddings=s_all_layer_embeddings,
                                gold_reference=gold_reference_binary)
s_all = pd.DataFrame(s_all)
s_all['embedding_level'] = ['sentence_all'] * len(s_all)
s_all['layer'] = list(range(s_all.shape[0]))


# Probing experiment sentence image 
s_img = probe.probing_experiment(layer_embeddings=s_img_layer_embeddings,
                                    gold_reference=gold_reference_binary)

s_img = pd.DataFrame(s_img)
s_img['embedding_level'] = ['sentence_img'] * len(s_img)
s_img['layer'] = list(range(s_img.shape[0]))

# Probing experiment sentence text 
s_text = probe.probing_experiment(layer_embeddings=s_text_layer_embeddings,
                                    gold_reference=gold_reference_binary)
s_text = pd.DataFrame(s_text)
s_text['embedding_level'] = ['sentence_text'] * len(s_text)
s_text['layer'] = list(range(s_text.shape[0]))

# Probing experiment token all 
t_all = probe.probing_experiment(layer_embeddings=t_all_layer_embeddings,
                                    gold_reference=gold_reference_binary)
t_all = pd.DataFrame(t_all)
t_all['embedding_level'] = ['token_all'] * len(t_all)
t_all['layer'] = list(range(t_all.shape[0]))

# Probing experiment token image 
t_img = probe.probing_experiment(layer_embeddings=t_img_layer_embeddings,
                                    gold_reference=gold_reference_binary)
t_img = pd.DataFrame(t_img)
t_img['embedding_level'] = ['token_img'] * len(t_img)
t_img['layer'] = list(range(t_img.shape[0]))

# Probing experiment token text 
t_text = probe.probing_experiment(layer_embeddings=t_text_layer_embeddings,
                                    gold_reference=gold_reference_binary)
t_text = pd.DataFrame(t_text)
t_text['embedding_level'] = ['token_text'] * len(t_text)
t_text['layer'] = list(range(t_text.shape[0]))

probe_result_df = pd.concat([s_all, s_img, s_text, t_all, t_img, t_text])

probe_result_df.to_csv(f"{results_path}/probing_results_{'_'.join(task)}_{'_'.join(old_target)}_to_{new_target}_{nobjects}.csv")
end = time.time()

print(f"Probe experiment completed, time taken {end - start}")
