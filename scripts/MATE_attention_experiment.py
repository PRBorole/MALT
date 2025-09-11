from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
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
model_name = 'llava-v1.6-mistral-7b' # gemma3-4B, Qwen2.5-VL-7B, llava_1.5_7b, llava-v1.6-mistral-7b, llava-v1.6-vicuna-7b
check_tril = False # Check if attention is uni or bidirectional

image_dir_path = './../datasets/MATE-dev/img/'
ds_main = load_jsonl_file('./../datasets/MATE-dev/mm_0shot_llava_hfllava_1.5_7b_hf.jsonl')
mate_df = pd.read_csv('./data/mate_df.csv',index_col=0)

task_dict = {
    'color': ['gray', 'yellow', 'red', 'blue', 'green'],
    'material': ['rubber', 'metal'],
    'shape': ['cone', 'cylinder', 'cube'],
    'size':['0.35', '0.351', '0.7', '0.701']
}

prompt_type='complex'
mode = 'image_and_text' # 'image', 'text', 'image_and_text',
nobjects = 7
task = ['color', 'shape']
target = ['gray', 'cylinder']

results_path = f'./results/llava1.6_mistral/attention/'
batch_inference_mode = 0 # get predicitons or not
attn_implementation = 'eager' #eager, sdpa, flash_attention_2, flash_attention_3

# Create probe dataset 
relevant_idx = mate_df[mate_df['object_count'] == nobjects]['idx'].to_list()
pos_idx = mate_df[(mate_df['object_count'] == nobjects) & 
                  (mate_df[('_'.join(task))].apply(lambda x: '_'.join(target) in x))]['idx'].to_list()


if len(pos_idx)>len(relevant_idx)/2:
    neg_idx = [i for i in relevant_idx if i not in pos_idx]
    pos_idx = random.sample(pos_idx, len(neg_idx))
else:
    neg_idx = random.sample([i for i in relevant_idx if i not in pos_idx], len(pos_idx))

data_idx = pos_idx + neg_idx
data = [ds_main[idx].copy() for idx in data_idx]
gold_reference_binary = [1]*len(pos_idx) + [0]*len(neg_idx)

# Load the model
model, processor, device = load_model_and_processor(
    model_name=model_name, 
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    attn_implementation=attn_implementation
)


probe = LinearProbe(
    model=model,
    nobjects=nobjects,
    task=task,
    mode=mode,
    target=target,
    prompt_type=prompt_type
)

# data = data[:1]
data = ds_main[:1]

for idx in range(len(data)):
    data[idx]['prompt'] = probe.get_prompt(data[idx])

images, prompts, inputs = process_input_data(ds=data,  
                                    image_dir_path=image_dir_path,
                                    processor=processor,
                                    device=device,
                                    model_name=model_name)

########### TEXT TOKENS FROM VLM 
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True, use_cache=False, )


attentions_vlm = [i.detach().cpu().to(dtype=torch.float16) for i in outputs.attentions]
nlayers = len(attentions_vlm)
nheads = attentions_vlm[0].shape[1]


if 'molmo' in model_name:
    image_tokens = np.where((inputs['input_ids']==processor.special_token_ids['<im_patch>']).cpu().detach().numpy().reshape(-1).tolist())[0]
else:
    image_tokens = np.where((inputs['input_ids']==model.config.image_token_id).cpu().detach().numpy().reshape(-1).tolist())[0]

# image_tokens = np.where((inputs['input_ids']==model.config.image_token_id).cpu().detach().numpy().reshape(-1).tolist())[0]
text_tokens = np.where(
                [False 
                if i in list(processor.tokenizer.all_special_ids) else True 
                for i in inputs['input_ids'][0].detach().cpu().tolist()]
            )[0]

# attn_vlm = attentions_vlm[30][0][0, text_tokens.tolist()].numpy().astype(np.float32)

########### ONLY TEXT NO IMAGE TOKENS
if 'molmo' in model_name:
    inputs_text_only = processor.process(images=None, text=prompts[0], padding=True, return_tensors="pt")
    inputs_text_only = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs_text_only.items()}

    text_only_tokens = np.where([False 
                if i in list(processor.tokenizer.all_special_ids) else True 
                for i in inputs_text_only['input_ids'].detach().cpu().tolist()]
            )[0]
else:
    inputs_text_only = processor(images=None, text=prompts, padding=True, return_tensors="pt")
    inputs_text_only = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs_text_only.items()}

    text_only_tokens = np.where([False 
                if i in list(processor.tokenizer.all_special_ids) else True 
                for i in inputs_text_only['input_ids'][0].detach().cpu().tolist()]
            )[0]

with torch.no_grad():
    # outputs = model(**{k:inputs[k][idx:idx+1] for k in inputs.keys()}, output_hidden_states=True)
    if len(inputs_text_only)==1:
        outputs = model(input_ids=inputs_text_only['input_ids'].reshape(1,-1), output_attentions=True, use_cache=False, output_hidden_states=False)
    else:
        outputs = model(**inputs_text_only, output_attentions=True, use_cache=False, output_hidden_states=False)

attentions_llm = [i.detach().cpu().to(dtype=torch.float16) for i in outputs.attentions]

# attn_llm = attentions_llm[30][0][0, text_only_tokens.tolist()].numpy().astype(np.float32)


# calculate entropy
vlm_entropy = np.zeros((nlayers, nheads))
llm_entropy = np.zeros((nlayers, nheads))
cross_modal_entropy = np.zeros((nlayers, nheads))
diff_entropy = np.zeros((nlayers, nheads))


for idx, layer in tqdm(enumerate(range(nlayers)), desc='idx'):
    for jdx, head in enumerate(range(nheads)):
        # attn = attentions[layer][0][head, image_tokens.tolist()][:,image_tokens.tolist()].numpy().astype(np.float32)
        # if np.allclose(attn, np.tril(attn)):
        #     attn = attn + attn.T - np.eye(attn.shape[0])*attn.diagonal() #symmetric attn
        # img_img_norm[idx][jdx] = np.linalg.norm(attn, ord='fro')/np.sqrt(attn.shape[0]*attn.shape[1])
        # img_img_entropy[idx][jdx] = get_matrix_entropy(attn)/np.sqrt(attn.shape[0]*attn.shape[1])
    
        # attn = attentions[layer, head, img_end:, img_end:]
        attn_vlm = attentions_vlm[layer][0][head, text_tokens.tolist()][:,text_tokens.tolist()].numpy().astype(np.float32)
        attn_cross = attentions_vlm[layer][0][head, text_tokens.tolist()][:,image_tokens.tolist()].numpy().astype(np.float32)
        attn_llm = attentions_llm[layer][0][head, text_only_tokens.tolist()][:,text_only_tokens.tolist()].numpy().astype(np.float32)

        vlm_entropy[idx][jdx] = attention_entropy(attn_vlm)
        llm_entropy[idx][jdx] = attention_entropy(attn_llm)
        cross_modal_entropy[idx][jdx] = attention_entropy(attn_cross)
        diff_entropy[idx][jdx] = llm_entropy[idx][jdx] - vlm_entropy[idx][jdx]
        
        # # attn = attentions[layer, head, img_end:, img_start:img_end]
        # attn = attentions[layer][0][head, text_tokens.tolist()][:,image_tokens.tolist()].numpy().astype(np.float32)
        # img_text_norm[idx][jdx] = np.linalg.norm(attn, ord='fro')/np.sqrt(attn.shape[0]*attn.shape[1])
        # if attn.sum()==0:
        #     attn = attn + 1e-12
        # img_text_entropy[idx][jdx] = get_matrix_entropy(attn)/np.sqrt(attn.shape[0]*attn.shape[1])


df1 = pd.DataFrame(vlm_entropy).T.melt()
df1['matrix'] = ['vlm']*len(df1)
df1['heads'] = list(range(nheads))*nlayers

df2 = pd.DataFrame(llm_entropy).T.melt()
df2['matrix'] = ['llm']*len(df2)
df2['heads'] = list(range(nheads))*nlayers

df3 = pd.DataFrame(diff_entropy).T.melt()
df3['matrix'] = ['diff']*len(df3)
df3['heads'] = list(range(nheads))*nlayers

df4 = pd.DataFrame(cross_modal_entropy).T.melt()
df4['matrix'] = ['cross_modal']*len(df4)
df4['heads'] = list(range(nheads))*nlayers

df = pd.concat([df1, df2, df3, df4])
df = df.rename(columns={'variable':'layers'})

df.to_csv(f"{results_path}/entropy_diff_{mode}_{'_'.join(target)}_{nobjects}.csv")

# df1 = pd.DataFrame(img_img_norm).T.melt()
# df1['attention_matrix'] = ['img_img_norm']*len(df1)
# df2 = pd.DataFrame(img_text_norm).T.melt()
# df2['attention_matrix'] = ['img_text_norm']*len(df2)
# df3 = pd.DataFrame(text_text_norm).T.melt()
# df3['attention_matrix'] = ['text_text_norm']*len(df3)

# pd.concat([df1, df2, df3]).to_csv(f"{results_path}/frobenius_attention_matrix_{mode}_{'_'.join(target)}_{nobjects}.csv")
# # pd.concat([df1, df2, df3]).to_csv(f"{results_path}/original_prompt_frobenius_attention_matrix.csv")

# df1 = pd.DataFrame(img_img_entropy).T.melt()
# df1['entropy'] = ['img_img_entropy']*len(df1)
# df2 = pd.DataFrame(img_text_entropy).T.melt()
# df2['entropy'] = ['img_text_entropy']*len(df2)
# df3 = pd.DataFrame(text_text_entropy).T.melt()
# df3['entropy'] = ['text_text_entropy']*len(df3)

# pd.concat([df1, df2, df3]).to_csv(f"{results_path}/entropy_attention_matrix_{mode}_{'_'.join(target)}_{nobjects}.csv")
# pd.concat([df1, df2, df3]).to_csv(f"{results_path}/original_prompt_entropy_attention_matrix.csv")

