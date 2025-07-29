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

task_dict = {
    'color': ['gray', 'yellow', 'red', 'blue', 'green'],
    'material': ['rubber', 'metal'],
    'shape': ['cone', 'cylinder', 'cube']
}


prompt_type = 'complex' # 'simple', 'complex'
mode = 'count' # 'count'
hide_object_name = True
results_path = f'./results/llava7B/linear_probing/probe_{mode}/{prompt_type}/hide_object_name_'
batch_inference_mode = 1 # get predicitons

nobjects = [3, 4, 5]
task = ['count']
target = [str(i) for i in nobjects]

# Create probe dataset 
mate_df_sub = mate_df.groupby("object_count").sample(frac=0.2)
probe_data_idx = mate_df_sub[mate_df_sub['object_count'].isin(nobjects)]['idx'].to_list()

probe_data = [ds_main[idx].copy() for idx in probe_data_idx]
gold_reference = mate_df_sub[mate_df_sub['object_count'].isin(nobjects)]['object_count'].to_list()

# Load the model
model, processor, device = load_model_and_processor(model_name='llava_1.5_7b')


# Set probe
probe = LinearProbe(
    model=model,
    nobjects=nobjects,
    task=task,
    mode=mode,
    target=target,
    prompt_type=prompt_type
)


for idx in range(len(probe_data)):
    prompt_text = probe.get_prompt(probe_data[idx])

    if hide_object_name:
        for nobjs in range(1,np.max(nobjects)+1):
            prompt_text = prompt_text.replace(f"Object_{nobjs}", "name")
    probe_data[idx]['prompt'] = prompt_text

flag = False
if flag:
    all_model_outputs = batch_inference(
        ds=probe_data, 
        image_dir_path=image_dir_path, 
        model=model, 
        processor=processor, 
        device=device, 
        batch_size=2, 
        max_new_tokens=50, 
        model_name='llava_1.5_7b'
    )

    predictions = [int(i.split('ASSISTANT:')[1].lstrip()) for i in all_model_outputs]


    f1_micro_vlm = f1_score(gold_reference, predictions, average='micro')
    precision_micro_vlm = precision_score(gold_reference, predictions, average='micro')
    recall_micro_vlm = recall_score(gold_reference, predictions,  average='micro')
    f1_macro_vlm = f1_score(gold_reference, predictions, average='macro')
    precision_macro_vlm = precision_score(gold_reference, predictions, average='macro')
    recall_macro_vlm = recall_score(gold_reference, predictions,  average='macro')

    # Save results
    pd.DataFrame(
        {'task': [task],
        'target': [target],
        'f1_micro': [f1_micro_vlm],
        'precision_micro': [precision_micro_vlm],
        'recall_micro': [recall_micro_vlm],
        'f1_macro': [f1_macro_vlm],
        'precision_macro': [precision_macro_vlm],
        'recall_macro': [recall_macro_vlm],

    }).to_csv(f"{results_path}vlm_results_{'_'.join(task)}_{'_'.join(target)}.csv", index=False)
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
# t_all_layer_embeddings = [None]*len(probe_data)
# t_img_layer_embeddings = [None]*len(probe_data)
# t_text_layer_embeddings = [None]*len(probe_data)

for idx in tqdm(range(len(probe_data))):
    model_embeddings = probe.get_layer_llava_embeddings(
        pixel_values=inputs['pixel_values'][idx:idx+1],
        input_ids=inputs['input_ids'][idx:idx+1],
        attention_mask=inputs['attention_mask'][idx:idx+1],
        mean_dim=1
    ) # For both level embeddings, 1 for sentence level, 2 for token level
    
    s_all_layer_embeddings[idx] = model_embeddings[0]
    # t_all_layer_embeddings[idx] = model_embeddings[1]
    s_img_layer_embeddings[idx] = model_embeddings[1]
    # t_img_layer_embeddings[idx] = model_embeddings[3]
    s_text_layer_embeddings[idx] = model_embeddings[2]
    # t_text_layer_embeddings[idx] = model_embeddings[5]

s_all_layer_embeddings = np.array(s_all_layer_embeddings) 
# t_all_layer_embeddings = np.array(t_all_layer_embeddings)
s_img_layer_embeddings = np.array(s_img_layer_embeddings)
# t_img_layer_embeddings = np.array(t_img_layer_embeddings)
s_text_layer_embeddings = np.array(s_text_layer_embeddings)
# t_text_layer_embeddings = np.array(t_text_layer_embeddings)


# # Probing experiment sentence all
# s_all = probe.probing_count_experiment(
#     layer_embeddings=s_all_layer_embeddings,
#     gold_reference=gold_reference
# )

# s_all = pd.DataFrame(s_all)
# s_all['embedding_level'] = ['sentence_all'] * len(s_all)
# s_all['layer'] = list(range(len(s_all)))

# Probing experiment sentence image 
s_img = probe.probing_count_experiment(
    layer_embeddings=s_img_layer_embeddings,
    gold_reference=gold_reference
)

s_img = pd.DataFrame(s_img)
s_img['embedding_level'] = ['sentence_img'] * len(s_img)
s_img.to_csv(f"{results_path}probing_results_s_img_{'_'.join(task)}_{'_'.join(target)}.csv")

# Probing experiment sentence text 
s_text = probe.probing_count_experiment(
    layer_embeddings=s_text_layer_embeddings,
    gold_reference=gold_reference
)
s_text = pd.DataFrame(s_text)
s_text['embedding_level'] = ['sentence_text'] * len(s_text)
s_text.to_csv(f"{results_path}probing_results_s_text_{'_'.join(task)}_{'_'.join(target)}.csv")

# # Probing experiment token all 
# t_all = probe.probing_count_experiment(
#     layer_embeddings=t_all_layer_embeddings,
#     gold_reference=gold_reference
# )
# t_all = pd.DataFrame(t_all)
# t_all['embedding_level'] = ['token_all'] * len(t_all)

# # Probing experiment token image 
# t_img = probe.probing_count_experiment(
#     layer_embeddings=t_img_layer_embeddings,
#     gold_reference=gold_reference
# )
# t_img = pd.DataFrame(t_img)
# t_img['embedding_level'] = ['token_img'] * len(t_img)

# # Probing experiment token text 
# t_text = probe.probing_count_experiment(
#     layer_embeddings=t_text_layer_embeddings,
#     gold_reference=gold_reference
# )
# t_text = pd.DataFrame(t_text)
# t_text['embedding_level'] = ['token_text'] * len(t_text)

probe_result_df = pd.concat([s_img, s_text])#, t_all, t_img, t_text])

probe_result_df.to_csv(f"{results_path}probing_results_{'_'.join(task)}_{'_'.join(target)}.csv")
end = time.time()

print(f"Probe experiment completed, time taken {end - start}")


# # Stratified split
# layer=3
# X_train, X_test, y_train, y_test = train_test_split(s_img_layer_embeddings[:,layer,:], 
#                                                     gold_reference, 
#                                                     test_size=0.2, 
#                                                     random_state=42, 
#                                                     stratify=gold_reference)

# # Train logistic regression
# if len(set(gold_reference))>2:
#     clf = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
# else:
#     clf = LogisticRegression(max_iter=1000, random_state=42)
# clf.fit(X_train, y_train)

# # Predict probabilities and labels
# y_pred = clf.predict(X_test)
# y_prob = clf.predict_proba(X_test)[:, 1]