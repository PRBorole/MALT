# from datasets import load_dataset
# from transformers import AutoProcessor, AutoModelForCausalLM
# from transformers import LlavaForConditionalGeneration, AutoModelForImageTextToText
# import torch
# from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
# from tqdm import tqdm
# from PIL import Image
# from src.utils import *
# import pandas as pd
# import numpy as np
# from src.probing import *
# import time 
# import random
# import re
# import ast
# import argparse
# import sys
# import os

# # Get the path of the Python script
# current_dir = os.path.abspath(os.path.dirname(__file__))
# # Exclude script name at the end
# current_dir = os.path.split(current_dir)[0]+'/'
# print(current_dir)

# sys.path.append(current_dir)
# from src.utils import *
# current_dir = '/exports/csce/eddie/inf/groups/ajitha_project/piyush/MALT/MALT/'
# ROOT_DIR = '/exports/csce/eddie/inf/groups/ajitha_project/piyush/MALT/'
# HF_dir = ROOT_DIR+'/models/'
# if os.path.isdir(HF_dir):
#     print(f"HF cache directory exists at: {HF_dir}")
# else:
#     raise FileNotFoundError(f"Directory does not exist: {HF_dir}")

# os.environ["HF_HUB_CACHE"] = HF_dir # set your HF cache directory
# os.environ["HF_HOME"] = HF_dir # set your HF cache directory


random.seed(42)
start = time.time()

image_dir_path = './../data/MATE-dev/img/'
ds_main = load_jsonl_file('./../data/MATE-dev/mm_0shot_llava_hfllava_1.5_7b_hf.jsonl')
mate_df = pd.read_csv('./data/mate_df.csv',index_col=0)

task_dict = {
    'color': ['gray', 'yellow', 'red', 'blue', 'green'],
    'material': ['rubber', 'metal'],
    'shape': ['cone', 'cylinder', 'cube'],
    'size':['0.35', '0.351', '0.7', '0.701']
}

# parser = argparse.ArgumentParser(description="usage help",
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--prompt_type", help="simple or complex")
# parser.add_argument("--mode", help="image, text or image_and_text")
# parser.add_argument("--nobjects", help="number of object in scene")
# parser.add_argument("--task", help="color, shape, material or combinations of these")
# parser.add_argument("--target", help="targets for task color, shape, material or combinations of these")

# args = parser.parse_args()
# config_path = vars(args)

# prompt_type = args.prompt_type
# mode = args.mode
# nobjects = int(args.nobjects)
# task = args.task.split('_')
# target = args.target.split('_')

prompt_type = 'simple' # 'simple', 'complex'
mode = 'image' # 'image', 'text', 'image_and_text', 'count'
# nobjects = 3
# task = ['color']
# target = ['red']

nobjects = 7
task = ['color_shape']
target = ['gray_cylinder']
model_name = 'Qwen2.5-VL-3B'

model_path = current_dir+'/../models/'

results_path = f'./results/{model_name}/linear_probing/probe_{mode}/'
batch_inference_mode = 1 # get predicitons
attn_implementation = 'sdpa'

# Create probe dataset 
relevant_idx = mate_df[mate_df['object_count'] == nobjects]['idx'].to_list()
pos_idx = mate_df[(mate_df['object_count'] == nobjects) & 
                  (mate_df[('_'.join(task))].apply(lambda x: '_'.join(target) in x))]['idx'].to_list()


if len(pos_idx)>len(relevant_idx)/2:
    neg_idx = [i for i in relevant_idx if i not in pos_idx]
    pos_idx = random.sample(pos_idx, len(neg_idx))
else:
    neg_idx = random.sample([i for i in relevant_idx if i not in pos_idx], len(pos_idx))

probe_data_idx = pos_idx + neg_idx
probe_data = [ds_main[idx].copy() for idx in probe_data_idx]
gold_reference_binary = [1]*len(pos_idx) + [0]*len(neg_idx)

# Load the model
model, processor, device = load_model_and_processor(
    model_name=model_name, 
    model_path=HF_dir, 
    attn_implementation=attn_implementation,
    torch_dtype=torch.float16 if 'mistral' in model_name else 'auto'
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

all_model_outputs = batch_inference(ds=probe_data, 
                                    image_dir_path=image_dir_path, 
                                    model=model, 
                                    processor=processor, 
                                    device=device, 
                                    batch_size=2, 
                                    max_new_tokens=50, 
                                    model_name=model_name)

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


# _, _, inputs = process_input_data(
#     ds=probe_data,  
#     image_dir_path=image_dir_path,
#     processor=processor,
#     device=device,
#     model_name=model_name
# )



##### sentence & token level embedding
s_all_layer_embeddings = [None]*len(probe_data)
s_img_layer_embeddings = [None]*len(probe_data)
s_text_layer_embeddings = [None]*len(probe_data)
# s_special_layer_embeddings = [None]*len(probe_data)
t_all_layer_embeddings = [None]*len(probe_data)
t_img_layer_embeddings = [None]*len(probe_data)
t_text_layer_embeddings = [None]*len(probe_data)

for idx in tqdm(range(len(probe_data))):
    # model_embeddings = probe.get_layer_llava_embeddings(
    #     pixel_values=inputs['pixel_values'][idx:idx+1],
    #     input_ids=inputs['input_ids'][idx:idx+1],
    #     attention_mask=inputs['attention_mask'][idx:idx+1],
    #     mean_dim='both',   # For both level embeddings, 1 for sentence level, 2 for token level
    #     special_token_embeddings=False
    # ) 
    
    # s_all_layer_embeddings[idx] = model_embeddings['ax1_all_layer_embeddings']
    # t_all_layer_embeddings[idx] = model_embeddings['ax2_all_layer_embeddings']
    # s_img_layer_embeddings[idx] = model_embeddings['ax1_img_layer_embeddings']
    # t_img_layer_embeddings[idx] = model_embeddings['ax2_img_layer_embeddings']
    # s_text_layer_embeddings[idx] = model_embeddings['ax1_text_layer_embeddings']
    # t_text_layer_embeddings[idx] = model_embeddings['ax2_text_layer_embeddings']
    
    batch = ( # We always use batch size 1 here
        probe_data[idx:idx+1] if isinstance(probe_data, list)
        else  probe_data.select(range(idx, idx+1))
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
    
    s_all_layer_embeddings[idx] = model_embeddings.astype(np.float32).mean(axis=1) 
    s_img_layer_embeddings[idx] = model_embeddings[:,image_tokens,:].astype(np.float32).mean(axis=1) 
    s_text_layer_embeddings[idx] = model_embeddings[:,text_tokens,:].astype(np.float32).mean(axis=1) 

s_all_layer_embeddings = np.array(s_all_layer_embeddings) 
# t_all_layer_embeddings = np.array(t_all_layer_embeddings)
s_img_layer_embeddings = np.array(s_img_layer_embeddings)
# t_img_layer_embeddings = np.array(t_img_layer_embeddings)
s_text_layer_embeddings = np.array(s_text_layer_embeddings)
# t_text_layer_embeddings = np.array(t_text_layer_embeddings)


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

# # Probing experiment sentence special 
# s_special = probe.probing_experiment(layer_embeddings=s_special_layer_embeddings,
#                                      gold_reference=gold_reference_binary)
# s_special = pd.DataFrame(s_special)
# s_special['embedding_level'] = ['sentence_text'] * len(s_special)
# s_special['layer'] = list(range(s_special.shape[0]))

# # Probing experiment token all 
# t_all = probe.probing_experiment(layer_embeddings=t_all_layer_embeddings,
#                                     gold_reference=gold_reference_binary)
# t_all = pd.DataFrame(t_all)
# t_all['embedding_level'] = ['token_all'] * len(t_all)
# t_all['layer'] = list(range(t_all.shape[0]))

# # Probing experiment token image 
# t_img = probe.probing_experiment(layer_embeddings=t_img_layer_embeddings,
#                                     gold_reference=gold_reference_binary)
# t_img = pd.DataFrame(t_img)
# t_img['embedding_level'] = ['token_img'] * len(t_img)
# t_img['layer'] = list(range(t_img.shape[0]))

# # Probing experiment token text 
# t_text = probe.probing_experiment(layer_embeddings=t_text_layer_embeddings,
#                                     gold_reference=gold_reference_binary)
# t_text = pd.DataFrame(t_text)
# t_text['embedding_level'] = ['token_text'] * len(t_text)
# t_text['layer'] = list(range(t_text.shape[0]))

# probe_result_df = pd.concat([s_all, s_img, s_text, t_all, t_img, t_text])
probe_result_df = pd.concat([s_all, s_img, s_text])
# probe_result_df = pd.concat([s_all, s_img, s_text, s_special])

probe_result_df.to_csv(f"{results_path}/{prompt_type}/probing_results_{'_'.join(task)}_{'_'.join(target)}_{nobjects}.csv")
end = time.time()

print(f"Probe experiment completed, time taken {end - start}")

# with torch.no_grad():
#     inputs_embeds = model.get_input_embeddings()(inputs['input_ids'][2:3])

# with torch.no_grad():
#     image_features = model.get_image_features(
#         pixel_values=inputs['pixel_values'][2:3]
#         )
#     image_features = torch.cat(image_features, dim=0)

# special_image_mask = inputs['input_ids'][0:1] == model.config.image_token_id
# n_image_tokens = (special_image_mask).sum()
# special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

# if inputs_embeds[special_image_mask].numel() != image_features.numel():
#     n_image_features = image_features.shape[0] * image_features.shape[1]
#     raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")
# image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
# inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

# with torch.no_grad():
#     outputs = model.language_model(attention_mask=inputs['attention_mask'][2:3],
#                                                     inputs_embeds=inputs_embeds,
#                                                     output_hidden_states=True,
#                                                     output_attentions=True,
#                                                     padding=True,
#                                                     attn_implementation=attn_implementation,
#                                                     return_dict_in_generation=True)


