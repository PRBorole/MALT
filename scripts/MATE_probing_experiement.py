from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import LlavaForConditionalGeneration, AutoModelForImageTextToText
import torch
from tqdm import tqdm
from PIL import Image
from src.utils import *
import pandas as pd
import numpy as np
from torchinfo import summary
from src.probing import *
import time 

start = time.time()

nobjects = 3
task = 'shape' # 'color', 'material', 'shape'
mode = 'image' # 'image', 'text'

if task == 'color':
    target = 'yellow' #gray, yellow, red, blue, green
elif task == 'material':
    target = 'metal' # rubber, metal
elif task == 'shape':
    target = 'cylinder'# cone, cube, cylinder

image_dir_path = './../datasets/MATE-dev/img/'
ds = load_jsonl_file('./../datasets/MATE-dev/mm_0shot_llava_hfllava_1.5_7b_hf.jsonl')

# Filter dataset for a specific number of objects
ds = [data for data in ds if data['object_count'] == nobjects]
gold_reference = ['Yes' if target in set([o[task] for o in data['scene']['objects']]) else 'No' for data in ds]
gold_reference_binary = [1 if i=='Yes' else 0 for i in gold_reference]

# Load the model
model, processor, device = load_model_and_processor(model_name='llava_1.5_7b')

# Set probe
probe = LinearProbe(model=model,
                    nobjects=nobjects,
                    task=task,
                    mode=mode,
                    target=target)

# Get prompts
for i in range(len(ds)):
    ds[i]['prompt'] = probe.get_prompt(ds[i])


batch_inference_mode=0
if batch_inference_mode:
    all_model_outputs = batch_inference(ds=ds, 
                                        image_dir_path=image_dir_path, 
                                        model=model, 
                                        processor=processor, 
                                        device=device, 
                                        batch_size=8, 
                                        max_new_tokens=50, 
                                        model_name='llava_1.5_7b')

    predictions = [i.split('ASSISTANT:')[1].lstrip() for i in all_model_outputs]

    mismatches = np.sum([1 if i.lower()!=j.lower() else 0 for i,j in zip(predictions,gold_reference)])

    print(mismatches, "mismatches out of", len(predictions))


_, _, inputs = process_input_data(ds=ds,  
                                    image_dir_path=image_dir_path,
                                    processor=processor,
                                    device=device,
                                    model_name='llava_1.5_7b')

##### sentence & token level embedding
s_all_layer_embeddings = [None]*len(ds)
s_img_layer_embeddings = [None]*len(ds)
s_text_layer_embeddings = [None]*len(ds)
t_all_layer_embeddings = [None]*len(ds)
t_img_layer_embeddings = [None]*len(ds)
t_text_layer_embeddings = [None]*len(ds)

for idx in tqdm(range(len(ds))):
    model_embeddings = probe.get_layer_llava_embeddings(pixel_values=inputs['pixel_values'][idx:idx+1],
                                                        input_ids=inputs['input_ids'][idx:idx+1],
                                                        attention_mask=inputs['attention_mask'][idx:idx+1],
                                                        mean_dim='both') # For both level embeddings, 1 for sentence level, 2 for token level
    
    s_all_layer_embeddings[idx] = model_embeddings[0]
    t_all_layer_embeddings[idx] = model_embeddings[1]
    s_img_layer_embeddings[idx] = model_embeddings[2]
    t_img_layer_embeddings[idx] = model_embeddings[3]
    s_text_layer_embeddings[idx] = model_embeddings[4]
    t_text_layer_embeddings[idx] = model_embeddings[5]

s_all_layer_embeddings = np.array(s_all_layer_embeddings)
t_all_layer_embeddings = np.array(t_all_layer_embeddings)
s_img_layer_embeddings = np.array(s_img_layer_embeddings)
t_img_layer_embeddings = np.array(t_img_layer_embeddings)
s_text_layer_embeddings = np.array(s_text_layer_embeddings)
t_text_layer_embeddings = np.array(t_text_layer_embeddings)

# Probing experiment sentence all 

s_all = probe.probing_experiment(layer_embeddings=s_all_layer_embeddings,
                                    gold_reference_binary=gold_reference_binary)
s_all = pd.DataFrame(s_all)
s_all['embedding_level'] = ['sentence_all'] * len(s_all)
s_all['layer'] = list(range(model.config.text_config.num_hidden_layers+1))

# Probing experiment sentence image 
s_img = probe.probing_experiment(layer_embeddings=s_img_layer_embeddings,
                                    gold_reference_binary=gold_reference_binary)
s_img = pd.DataFrame(s_img)
s_img['embedding_level'] = ['sentence_img'] * len(s_img)
s_img['layer'] = list(range(model.config.text_config.num_hidden_layers+1))

# Probing experiment sentence text 
s_text = probe.probing_experiment(layer_embeddings=s_text_layer_embeddings,
                                    gold_reference_binary=gold_reference_binary)
s_text = pd.DataFrame(s_text)
s_text['embedding_level'] = ['sentence_text'] * len(s_text)
s_text['layer'] = list(range(model.config.text_config.num_hidden_layers+1))

# Probing experiment token all 
t_all = probe.probing_experiment(layer_embeddings=t_all_layer_embeddings,
                                    gold_reference_binary=gold_reference_binary)
t_all = pd.DataFrame(t_all)
t_all['embedding_level'] = ['token_all'] * len(t_all)
t_all['layer'] = list(range(model.config.text_config.num_hidden_layers+1))

# Probing experiment token image 
t_img = probe.probing_experiment(layer_embeddings=t_img_layer_embeddings,
                                    gold_reference_binary=gold_reference_binary)
t_img = pd.DataFrame(t_img)
t_img['embedding_level'] = ['token_img'] * len(t_img)
t_img['layer'] = list(range(model.config.text_config.num_hidden_layers+1))

# Probing experiment token text 
t_text = probe.probing_experiment(layer_embeddings=t_text_layer_embeddings,
                                    gold_reference_binary=gold_reference_binary)
t_text = pd.DataFrame(t_text)
t_text['embedding_level'] = ['token_text'] * len(t_text)
t_text['layer'] = list(range(model.config.text_config.num_hidden_layers+1))

df = pd.concat([s_all, s_img, s_text, t_all, t_img, t_text])
df['task'] = [task] * len(df)
df['target'] = [target] * len(df)
                
df.to_csv(f"./results/llava7B/linear_probing/{mode}/probing_results_{task}_{target}_{nobjects}.csv")

end=time.time()
print(f"Time taken for probing experiment: {end-start}")
print("Probing experiment completed and results saved to CSV file.")

