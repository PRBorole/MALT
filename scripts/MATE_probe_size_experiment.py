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
from torchinfo import summary
from src.probing import *
import time 
import random
import re
import ast

random.seed(42)
start = time.time()

image_dir_path = './../datasets/MATE-dev/img/'
ds_main = load_jsonl_file('./../datasets/MATE-dev/mm_0shot_llava_hfllava_1.5_7b_hf.jsonl')
mate_df = pd.read_csv('./data/mate_df.csv',index_col=0)
results_path = './results/llava7B/linear_probing/probe_size/'

task_dict = {'color': ['gray', 'yellow', 'red', 'blue', 'green'],
             'material': ['rubber', 'metal'],
             'shape': ['cone', 'cylinder', 'cube']}


prompt_type = 'complex' # 'simple', 'complex'
mode = 'image' # 'image', 'text', 'image_and_text',
batch_inference_mode = 1 # get predicitons

nobjects = 3
k_reps = 5
task = ['color']
target = ['red']

# Create probe dataset 
relevant_idx = mate_df[mate_df['object_count'] == nobjects]['idx'].to_list()
pos_idx = mate_df[(mate_df['object_count'] == nobjects) & 
                  (mate_df[('_'.join(task))].apply(lambda x: '_'.join(target) in x))]['idx'].to_list()
neg_idx = random.sample([i for i in relevant_idx if i not in pos_idx], len(pos_idx))
probe_data_idx = pos_idx + neg_idx
probe_data = [ds_main[idx].copy() for idx in probe_data_idx]
gold_reference_binary = [1]*len(pos_idx) + [0]*len(neg_idx)

# Set dataset sizes for experiements
datasize_ls = list(range(25,150,25)) + [len(gold_reference_binary)]

# Load the model
model, processor, device = load_model_and_processor(model_name='llava_1.5_7b')


# Set probe
probe = LinearProbe(model=model,
                    nobjects=nobjects,
                    task=task,
                    mode=mode,
                    target=target,
                    prompt_type=prompt_type)


for idx in range(len(probe_data)):
    probe_data[idx]['prompt'] = probe.get_prompt(probe_data[idx])


all_model_outputs = batch_inference(ds=probe_data, 
                                    image_dir_path=image_dir_path, 
                                    model=model, 
                                    processor=processor, 
                                    device=device, 
                                    batch_size=2, 
                                    max_new_tokens=50, 
                                    model_name='llava_1.5_7b')

predictions = [i.split('ASSISTANT:')[1].lstrip() for i in all_model_outputs]


f1_vlm = f1_score(gold_reference_binary, [1 if i.lower()=='yes' else 0 for i in predictions])
precision_vlm = precision_score(gold_reference_binary, [1 if i.lower()=='yes' else 0 for i in predictions])
recall_vlm = recall_score(gold_reference_binary, [1 if i.lower()=='yes' else 0 for i in predictions])
# Save results
pd.DataFrame({'task': [task],
                'target': [target],
                'f1': [f1_vlm],
                'precision': [precision_vlm],
                'recall': [recall_vlm]}).to_csv(f"{results_path}/{prompt_type}/vlm_results_{'_'.join(task)}_{'_'.join(target)}_{nobjects}.csv", index=False)
print("Probing experiment completed and results saved to CSV file.")


_, _, inputs = process_input_data(ds=probe_data,  
                                    image_dir_path=image_dir_path,
                                    processor=processor,
                                    device=device,
                                    model_name='llava_1.5_7b')

##### sentence & token level embedding
s_all_layer_embeddings = [None]*len(probe_data)
s_img_layer_embeddings = [None]*len(probe_data)
s_text_layer_embeddings = [None]*len(probe_data)
t_all_layer_embeddings = [None]*len(probe_data)
t_img_layer_embeddings = [None]*len(probe_data)
t_text_layer_embeddings = [None]*len(probe_data)

for idx in tqdm(range(len(probe_data))):
    model_embeddings = probe.get_layer_llava_embeddings(pixel_values=inputs['pixel_values'][idx:idx+1],
                                                        input_ids=inputs['input_ids'][idx:idx+1],
                                                        attention_mask=inputs['attention_mask'][idx:idx+1],
                                                        mean_dim='both') # For both level embeddings, 1 for sentence level, 2 for token level
    
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

layers = list(range(0,33,7)) # Keep every 7th layer to get results for 33/7 ~ 5 layers


for datasize in tqdm(datasize_ls, desc='datasize'):
    probe_result_df = []
    for k in tqdm(range(k_reps),desc='k'):

        if datasize==len(gold_reference_binary):
            data_idx = list(range(len(gold_reference_binary)))
        else:
            # Create subset data
            halfpoint = int(len(gold_reference_binary)/2)
            data_idx = random.sample(list(range(halfpoint)), datasize) # subset pos 
            data_idx = data_idx + random.sample(list(range(halfpoint,len(gold_reference_binary))), datasize) # subset neg 

        gold_reference_binary_subset = np.array(gold_reference_binary)[data_idx]

        assert gold_reference_binary_subset.sum()/len(gold_reference_binary_subset)==0.5, f"subsample if not balanced, {gold_reference_binary_subset.sum()/len(gold_reference_binary_subset)}"
    
        s_all_layer_embeddings_subset = s_all_layer_embeddings[data_idx]
        t_all_layer_embeddings_subset = t_all_layer_embeddings[data_idx]
        s_img_layer_embeddings_subset = s_img_layer_embeddings[data_idx]
        t_img_layer_embeddings_subset = t_img_layer_embeddings[data_idx]
        s_text_layer_embeddings_subset = s_text_layer_embeddings[data_idx]
        t_text_layer_embeddings_subset = t_text_layer_embeddings[data_idx]

        # Probing experiment sentence all
        s_all = probe.probing_experiment(layer_embeddings=s_all_layer_embeddings_subset,
                                        gold_reference_binary=gold_reference_binary_subset,
                                        layers=layers)
        s_all = pd.DataFrame(s_all)
        s_all['embedding_level'] = ['sentence_all'] * len(s_all)
        s_all['layer'] = layers

        # Probing experiment sentence image 
        s_img = probe.probing_experiment(layer_embeddings=s_img_layer_embeddings_subset,
                                            gold_reference_binary=gold_reference_binary_subset,
                                            layers=layers)
        s_img = pd.DataFrame(s_img)
        s_img['embedding_level'] = ['sentence_img'] * len(s_img)
        s_img['layer'] = layers

        # Probing experiment sentence text 
        s_text = probe.probing_experiment(layer_embeddings=s_text_layer_embeddings_subset,
                                            gold_reference_binary=gold_reference_binary_subset,
                                            layers=layers)
        s_text = pd.DataFrame(s_text)
        s_text['embedding_level'] = ['sentence_text'] * len(s_text)
        s_text['layer'] = layers

        # Probing experiment token all 
        t_all = probe.probing_experiment(layer_embeddings=t_all_layer_embeddings_subset,
                                            gold_reference_binary=gold_reference_binary_subset,
                                            layers=layers)
        t_all = pd.DataFrame(t_all)
        t_all['embedding_level'] = ['token_all'] * len(t_all)
        t_all['layer'] = layers

        # Probing experiment token image 
        t_img = probe.probing_experiment(layer_embeddings=t_img_layer_embeddings_subset,
                                            gold_reference_binary=gold_reference_binary_subset,
                                            layers=layers)
        t_img = pd.DataFrame(t_img)
        t_img['embedding_level'] = ['token_img'] * len(t_img)
        t_img['layer'] = layers

        # Probing experiment token text 
        t_text = probe.probing_experiment(layer_embeddings=t_text_layer_embeddings_subset,
                                            gold_reference_binary=gold_reference_binary_subset,
                                            layers=layers)
        t_text = pd.DataFrame(t_text)
        t_text['embedding_level'] = ['token_text'] * len(t_text)
        t_text['layer'] = layers

        probe_result_df_ = pd.concat([s_all, s_img, s_text, t_all, t_img, t_text])
        probe_result_df_['k'] = [k]*len(probe_result_df_)

        probe_result_df = probe_result_df + [probe_result_df_]

    probe_result_df = pd.concat(probe_result_df)
    probe_result_df.to_csv(f"{results_path}/{prompt_type}/probing_results_{'_'.join(task)}_{'_'.join(target)}_{nobjects}_datasize_{len(gold_reference_binary_subset)}.csv")
