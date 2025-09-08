# feedback_refinement.py
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import LlavaForConditionalGeneration, AutoModelForImageTextToText
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

class TopDownFuse(nn.Module):
    """
    Per-layer gated residual that injects top-down states (from layer n+1, previous iteration)
    into the current layer's input.
    """
    def __init__(self):
        super().__init__()
        # self.hid = hidden_size
        # mid = bottleneck if bottleneck > 0 else hidden_size // 2
        # self.norm_h  = nn.LayerNorm(hidden_size, eps=1e-5, dtype=torch.float16)
        # self.norm_td = nn.LayerNorm(hidden_size, eps=1e-5, dtype=torch.float16)
        # self.proj_td = nn.Linear(hidden_size, hidden_size, bias=True, dtype=torch.float16)
        # self.gate = nn.Sequential(
        #     nn.Linear(2 * hidden_size, mid, bias=True, dtype=torch.float16),
        #     nn.SiLU(),
        #     nn.Linear(mid, hidden_size, bias=True, dtype=torch.float16),
        #     nn.Sigmoid(),
        # )

    def forward(self, h: torch.Tensor, td: Optional[torch.Tensor]) -> torch.Tensor:
        if td is None:
            return h
        # h_n  = self.norm_h(h)
        # td_n = self.norm_td(td)
        # g = self.gate(torch.cat([h_n, td_n], dim=-1))       # (B, T, H)
        # td_proj = self.proj_td(td)
        return (h + td)/2
        # return h + g * td_proj                               # gated residual inject


class RecurrentRefinementLlama(nn.Module):
    """
    Wraps a HF LlamaModel (the 'model' submodule inside LlamaForCausalLM / LLaVA).
    Runs K refinement iterations. On iteration k>0, for each layer n we fuse in the
    previous iteration's (n+1)-th layer hidden states as top-down input.
    """
    def __init__(self, llama_model, num_iters: int = 2):
        super().__init__()
        self.llama = llama_model                 # e.g., llava_model.model  (HF LlamaModel)
        L = len(llama_model.layers)
        self.num_layers = L
        self.num_iters = num_iters
        self.fuse = nn.ModuleList([TopDownFuse() for _ in range(L)])

    @torch.no_grad()
    def _embed_inputs(self, **fw) -> Dict[str, Any]:
        """
        Build the initial hidden_states and masks the same way LlamaModel.forward does,
        but stop before running any decoder layers. We stay version-agnostic.
        """
        # Accept either input_ids or inputs_embeds (LLaVA often uses inputs_embeds)
        if fw.get("inputs_embeds", None) is None:
            input_ids = fw["input_ids"]
            inputs_embeds = self.llama.embed_tokens(input_ids)
        else:
            inputs_embeds = fw["inputs_embeds"]
        attention_mask = fw.get("attention_mask", None)
        position_ids = fw.get("position_ids", None)
        return dict(hidden_states=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids)

    def _run_once(self, hidden_states, attention_mask,position_ids,position_embeddings,
                  topdown_per_layer: Optional[List[Optional[torch.Tensor]]] = None,
                  output_all_layers: bool = True) -> List[torch.Tensor]:
        """
        Run the decoder stack once. If topdown_per_layer is provided, it must be a list
        with length == num_layers, where entry n is the top-down tensor for layer n
        (commonly previous-iteration layer n+1 output), else None.
        Returns per-layer outputs (post-attn+MLP residual), length L.
        """
        per_layer: List[torch.Tensor] = []
        x = hidden_states
        for n, layer in enumerate(self.llama.layers):
            td = None if topdown_per_layer is None else topdown_per_layer[n]
            x = self.fuse[n](x, td)                     # inject feedback BEFORE the real layer
            # standard LlamaDecoderLayer forward; disable cache in refinement
            x = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                position_embeddings=position_embeddings,
                output_attentions=False,
                use_cache=False,
            )[0]  # first element is hidden_states
            if output_all_layers:
                per_layer.append(x)
        return per_layer
        
    
        # with torch.no_grad():
        #     per_layer = self.llama(attention_mask=attention_mask,
        #                             inputs_embeds=inputs_embeds,
        #                             output_hidden_states=True,
        #                             output_attentions=False,
        #                             padding=True,
        #                             return_dict=True)
        # return per_layer['hidden_states']

    def forward(self, *,  # keyword-only for clarity
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                cache_position: Optional[torch.FloatTensor] = None,
                past_key_values: Optional[torch.FloatTensor] = None,
                return_all_iters: bool = False) -> Dict[str, torch.Tensor]:
        """
        Run K refinement iterations. Returns final hidden states (post final_norm).
        If return_all_iters=True, also returns per-iteration layer stacks for analysis.
        """
        # Prepare inputs like LlamaModel
        fw = dict(input_ids=input_ids, attention_mask=attention_mask,
                  position_ids=position_ids, inputs_embeds=inputs_embeds)
        prep = self._embed_inputs(**fw)
        x = prep["hidden_states"]
        attn_mask = prep["attention_mask"]

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.llama.rotary_emb(x, position_ids)

        # Iteration 0: normal bottom-up pass (no feedback)
        with torch.no_grad():                  # keep iteration-0 as the "previous" snapshot
            per_layer_prev = self._run_once(x, attn_mask, position_ids, position_embeddings=position_embeddings, topdown_per_layer=None)

        per_iter_states = [per_layer_prev] if return_all_iters else None

        # Iterations 1..K-1: recurrent refinement with top-down from previous iteration
        for idx in range(0, self.num_iters):
            # Align: for layer n we use previous iteration's layer (n+1) output.
            topdown = [per_layer_prev[n + 1].detach() if (n + 1) < self.num_layers else None
                       for n in range(self.num_layers)]
            per_layer_curr = self._run_once(x, attn_mask, position_ids, position_embeddings=position_embeddings, topdown_per_layer=topdown)
            per_layer_prev = per_layer_curr

            if return_all_iters:
                per_iter_states.append(per_layer_curr)
            # Update x for next iteration start = output of last layer
            x = per_layer_curr[-1].detach()
        
        # Final norm (same as LlamaModel)
        # final = self.llama.norm(per_layer_prev[-1])
        final = per_layer_prev[-1]
        out = {"last_hidden_state": final}
        if return_all_iters:
            out["per_iter_layer_states"] = per_iter_states
        return out



################################################################################### code VLM
# random.seed(42)
# image_dir_path = './../datasets/MATE-dev/img/'
# ds_main = load_jsonl_file('./../datasets/MATE-dev/mm_0shot_llava_hfllava_1.5_7b_hf.jsonl')
# mate_df = pd.read_csv('./data/mate_df.csv',index_col=0)
# linear_probe = True

# attn_implementation = 'sdpa'
# nsamples = 100

# # Load the model
# model, processor, device = load_model_and_processor(
#     model_name='llava_1.5_7b', 
#     attn_implementation=attn_implementation
# )
# model.eval()
# ds_main = ds_main[:nsamples]


# images, prompts, inputs = process_input_data(ds=ds_main,  
#                                             image_dir_path=image_dir_path,
#                                             processor=processor,
#                                             device=device,
#                                             model_name='llava_1.5_7b')

# predictions = {}
# # for num_iters in tqdm(range(10,-1,-1)):
# for num_iters in [0,2, 5]:
#     decoded = []
#     for idx in range(len(ds_main)):
#         input_ids = inputs['input_ids'][idx:idx+1].to(device)
#         attention_mask = inputs['attention_mask'][idx:idx+1].to(device,dtype=torch.float16)
#         pixel_values = inputs['pixel_values'][idx:idx+1].to(device)

#         vision_feature_layer = None
#         vision_feature_select_strategy = None
#         vision_feature_layer = (vision_feature_layer if vision_feature_layer is not None else model.config.vision_feature_layer)
#         vision_feature_select_strategy = (vision_feature_select_strategy
#                                             if vision_feature_select_strategy is not None
#                                             else model.config.vision_feature_select_strategy)

#         with torch.no_grad():
#             inputs_embeds = model.get_input_embeddings()(input_ids)

#         with torch.no_grad():
#             image_features = model.get_image_features(
#                 pixel_values=pixel_values,
#                 vision_feature_layer=vision_feature_layer,
#                 vision_feature_select_strategy=vision_feature_select_strategy
#                 )
#             image_features = torch.cat(image_features, dim=0)

#         special_image_mask = input_ids == model.config.image_token_id
#         n_image_tokens = (special_image_mask).sum()
#         special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

#         if inputs_embeds[special_image_mask].numel() != image_features.numel():
#             n_image_features = image_features.shape[0] * image_features.shape[1]
#             raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")
#         image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
#         inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

#         llama_backbone = model.language_model
#         refiner = RecurrentRefinementLlama(llama_backbone, num_iters=num_iters).to(device)


#         with torch.no_grad():
#             outputs = refiner(attention_mask=attention_mask,
#                                 position_ids=None,
#                                 inputs_embeds=inputs_embeds)
            
#         outputs = outputs['last_hidden_state']
#         # Convert to logits with the LM head
#         with torch.no_grad():
#             logits = model.lm_head(outputs[:, -1:, :])  # shape: (B, 1, vocab_size)

#         # Choose next token (greedy here; you can replace with sampling logic)
#         next_token_id = torch.argmax(logits, dim=-1)

#         # We'll store generated tokens here
#         generated = input_ids.clone()
#         past_key_values = None

#         # Append token to the generated sequence
#         generated = torch.cat([generated, next_token_id], dim=-1)
#         attention_mask = torch.cat([attention_mask,torch.tensor([[1]],device=device)],dim=-1)

#         # Generate exactly <> new tokens
#         for step in range(50):
#             with torch.no_grad():
#                 outputs = model(
#                     pixel_values=pixel_values,
#                     attention_mask=attention_mask,
#                     input_ids=generated,       # feed only the last token
#                     # past_key_values=past_key_values,   # speed via caching
#                     # use_cache=True,                    # ensure caching is enabled
#                     output_hidden_states=True,         # to get hidden states
#                     return_dict=True
#                 )
            
#             # outputs.decoder_hidden_states is a tuple: (embeddings, layer1, ..., last_layer)
#             decoder_hidden_states = outputs.hidden_states

#             # Extract the hidden state of the last layer, last token
#             last_hidden = decoder_hidden_states[-1][:, -1:, :]  # shape: (B, 1, hidden_size)

#             # Convert to logits with the LM head
#             with torch.no_grad():
#                 logits = model.lm_head(last_hidden)  # shape: (B, 1, vocab_size)

#             # Choose next token (greedy here; you can replace with sampling logic)
#             next_token_id = torch.argmax(logits, dim=-1)

#             # Append token to the generated sequence
#             generated = torch.cat([generated, next_token_id], dim=-1)
#             attention_mask = torch.cat([attention_mask,torch.tensor([[1]],device=device)],dim=-1)

#             if next_token_id.item() == processor.tokenizer.eos_token_id:
#                 break

#         # Decode the full generated sequence
#         decoded = decoded + [processor.tokenizer.decode(generated[0], skip_special_tokens=True)]

#     predictions[num_iters] = [i.split('ASSISTANT')[1] for i in decoded]
    
# predictions

# for idx,i in enumerate(predictions[5]):
#     print(f"gold reference is {ds_main[idx]['gold_reference']} and prediction is ", i.replace('\n',''))

# for k,v in predictions.items():
#     y_pred = [1 if 'yes' in i.lower() else 0 for i in v]
#     accuracy = np.sum([1 if i==j else 0 for i,j in zip(y_pred, gold_reference)])/len(gold_reference)
#     print(k, accuracy)

# #################################################################################### code LP
# random.seed(42)
# image_dir_path = './../datasets/MATE-dev/img/'
# ds_main = load_jsonl_file('./../datasets/MATE-dev/mm_0shot_llava_hfllava_1.5_7b_hf.jsonl')
# mate_df = pd.read_csv('./data/mate_df.csv',index_col=0)
# linear_probe = True

# attn_implementation = 'sdpa'
# nsamples = 50

# # Load the model
# model, processor, device = load_model_and_processor(
#     model_name='llava_1.5_7b', 
#     attn_implementation=attn_implementation
# )
# model.eval()

# task_dict = {
#     'color': ['gray', 'yellow', 'red', 'blue', 'green'],
#     'material': ['rubber', 'metal'],
#     'shape': ['cone', 'cylinder', 'cube'],
#     'size':['0.35', '0.351', '0.7', '0.701']
# }

# prompt_type = 'complex' # 'simple', 'complex'
# mode = 'image' # 'image', 'text', 'image_and_text',
# nobjects = 10
# task = ['color','shape']
# target = ['gray', 'cylinder']



# #  Create probe dataset 
# relevant_idx = mate_df[mate_df['object_count'] == nobjects]['idx'].to_list()
# pos_idx = mate_df[(mate_df['object_count'] == nobjects) & 
#                 (mate_df[('_'.join(task))].apply(lambda x: '_'.join(target) in x))]['idx'].to_list()
# neg_idx = [i for i in relevant_idx if i not in pos_idx]

# neg_idx = random.sample(neg_idx, nsamples)
# pos_idx = random.sample(pos_idx, nsamples)

# probe_data_idx = pos_idx + neg_idx
# ds_main = [ds_main[idx].copy() for idx in probe_data_idx]
# gold_reference = [1]*len(pos_idx) + [0]*len(neg_idx)

# # Set probe
# probe = LinearProbe(model=model,
#                     nobjects=nobjects,
#                     task=task,
#                     mode=mode,
#                     target=target,
#                     prompt_type=prompt_type)


# for idx in range(len(ds_main)):
#     ds_main[idx]['prompt'] = probe.get_prompt(ds_main[idx])


# images, prompts, inputs = process_input_data(ds=ds_main,  
#                                             image_dir_path=image_dir_path,
#                                             processor=processor,
#                                             device=device,
#                                             model_name='llava_1.5_7b')

# predictions = {}
# for num_iters in tqdm(range(100,20,-10)):
#     decoded = []
#     for idx in range(len(ds_main)):
#         input_ids = inputs['input_ids'][idx:idx+1].to(device)
#         attention_mask = inputs['attention_mask'][idx:idx+1].to(device,dtype=torch.float16)
#         pixel_values = inputs['pixel_values'][idx:idx+1].to(device)

#         vision_feature_layer = None
#         vision_feature_select_strategy = None
#         vision_feature_layer = (vision_feature_layer if vision_feature_layer is not None else model.config.vision_feature_layer)
#         vision_feature_select_strategy = (vision_feature_select_strategy
#                                             if vision_feature_select_strategy is not None
#                                             else model.config.vision_feature_select_strategy)

#         with torch.no_grad():
#             inputs_embeds = model.get_input_embeddings()(input_ids)

#         with torch.no_grad():
#             image_features = model.get_image_features(
#                 pixel_values=pixel_values,
#                 vision_feature_layer=vision_feature_layer,
#                 vision_feature_select_strategy=vision_feature_select_strategy
#                 )
#             image_features = torch.cat(image_features, dim=0)

#         special_image_mask = input_ids == model.config.image_token_id
#         n_image_tokens = (special_image_mask).sum()
#         special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

#         if inputs_embeds[special_image_mask].numel() != image_features.numel():
#             n_image_features = image_features.shape[0] * image_features.shape[1]
#             raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")
#         image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
#         inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)



#         llama_backbone = model.language_model
#         refiner = RecurrentRefinementLlama(llama_backbone, num_iters=num_iters).to(device)


#         with torch.no_grad():
#             outputs = refiner(attention_mask=attention_mask,
#                                 position_ids=None,
#                                 inputs_embeds=inputs_embeds)
            
#         outputs = outputs['last_hidden_state']
#         # Convert to logits with the LM head
#         with torch.no_grad():
#             logits = model.lm_head(outputs[:, -1:, :])  # shape: (B, 1, vocab_size)

#         # Choose next token (greedy here; you can replace with sampling logic)
#         next_token_id = torch.argmax(logits, dim=-1)

#         # We'll store generated tokens here
#         generated = input_ids.clone()
#         past_key_values = None

#         # Append token to the generated sequence
#         generated = torch.cat([generated, next_token_id], dim=-1)
#         attention_mask = torch.cat([attention_mask,torch.tensor([[1]],device=device)],dim=-1)

#         # Generate exactly <> new tokens
#         for step in range(50):
#             with torch.no_grad():
#                 outputs = model(
#                     pixel_values=pixel_values,
#                     attention_mask=attention_mask,
#                     input_ids=generated,       # feed only the last token
#                     # past_key_values=past_key_values,   # speed via caching
#                     # use_cache=True,                    # ensure caching is enabled
#                     output_hidden_states=True,         # to get hidden states
#                     return_dict=True
#                 )
            
#             # outputs.decoder_hidden_states is a tuple: (embeddings, layer1, ..., last_layer)
#             decoder_hidden_states = outputs.hidden_states

#             # Extract the hidden state of the last layer, last token
#             last_hidden = decoder_hidden_states[-1][:, -1:, :]  # shape: (B, 1, hidden_size)

#             # Convert to logits with the LM head
#             with torch.no_grad():
#                 logits = model.lm_head(last_hidden)  # shape: (B, 1, vocab_size)

#             # Choose next token (greedy here; you can replace with sampling logic)
#             next_token_id = torch.argmax(logits, dim=-1)

#             # Append token to the generated sequence
#             generated = torch.cat([generated, next_token_id], dim=-1)
#             attention_mask = torch.cat([attention_mask,torch.tensor([[1]],device=device)],dim=-1)

#             if next_token_id.item() == processor.tokenizer.eos_token_id:
#                 break


#         # Decode the full generated sequence
#         decoded = decoded + [processor.tokenizer.decode(generated[0], skip_special_tokens=True)]

#     predictions[num_iters] = [i.split('ASSISTANT')[1] for i in decoded]
    
# predictions

# for k,v in predictions.items():
#     y_pred = [1 if 'yes' in i.lower() else 0 for i in v]
#     accuracy = np.sum([1 if i==j else 0 for i,j in zip(y_pred, gold_reference)])/len(gold_reference)
#     print(k, accuracy)

#################################################################################### code LRA

def LRA(M, r):
    '''Decomposes the matrix M using Singular Value Decomposition (SVD).'''
    if r=='asis':
        return M
    
    with torch.no_grad():
        U,S,V = torch.linalg.svd(M, full_matrices=False)
        U = U.detach().cpu()
        S = S.detach().cpu()
        V = V.detach().cpu()
        M = M.detach().cpu()


    # Adjust matrix shapes to ensure U (m x n), S (n x n), V (n x n):
    U = U[:, :S.shape[0]]  # Limit U to the number of singular values 
    S = torch.diag(S)  # Convert S (a vector) into a diagonal matrix
    if r<0:
        r = S.shape[0]+r
    print(S.shape, "rank: ", r)
    R = U[:, :r] @ S[:r, :r] @ V[:r, :]  # Use rank-reduced components for approximation 

    return R

def compare_matrices(M, R):
    '''Calculates the difference and relative difference between the original and 
        reconstructed matrix.'''
    D = M - R  # Element-wise difference
    M_norm = torch.linalg.norm(M, 'fro')  # Frobenius norm of original matrix
    D_norm = torch.linalg.norm(D, 'fro')  # Frobenius norm of difference matrix
    print(M_norm, D_norm)
    return D_norm / M_norm  # Calculate the relative difference ratio

# # Test the functions with sample values:
# R = LRA(M, 2)  # Decompose with SVD 

# diff_ratio1 = compare_matrices(M, R)
# print(f'Diff ratio 1: {diff_ratio1:.2f}')

random.seed(42)
image_dir_path = './../datasets/MATE-dev/img/'
ds_main = load_jsonl_file('./../datasets/MATE-dev/mm_0shot_llava_hfllava_1.5_7b_hf.jsonl')
mate_df = pd.read_csv('./data/mate_df.csv',index_col=0)
linear_probe = True

attn_implementation = 'sdpa'
nsamples = 5
# nrank = 'asis'
nrank = -1025


# # Load the model
# model, processor, device = load_model_and_processor(
#     model_name='llava_1.5_7b', 
#     attn_implementation=attn_implementation
# )
# model.eval()

task_dict = {
    'color': ['gray', 'yellow', 'red', 'blue', 'green'],
    'material': ['rubber', 'metal'],
    'shape': ['cone', 'cylinder', 'cube'],
    'size':['0.35', '0.351', '0.7', '0.701']
}

prompt_type = 'complex' # 'simple', 'complex'
mode = 'image_and_text' # 'image', 'text', 'image_and_text',
nobjects = 3
task = ['color','shape']
target = ['gray', 'cylinder']



#  Create probe dataset 
relevant_idx = mate_df[mate_df['object_count'] == nobjects]['idx'].to_list()
pos_idx = mate_df[(mate_df['object_count'] == nobjects) & 
                (mate_df[('_'.join(task))].apply(lambda x: '_'.join(target) in x))]['idx'].to_list()
neg_idx = [i for i in relevant_idx if i not in pos_idx]

neg_idx = random.sample(neg_idx, nsamples)
pos_idx = random.sample(pos_idx, nsamples)

probe_data_idx = pos_idx + neg_idx
ds_main = [ds_main[idx].copy() for idx in probe_data_idx]
gold_reference = [1]*len(pos_idx) + [0]*len(neg_idx)

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

decoded = []
for idx in tqdm(range(len(ds_main))):
    input_ids = inputs['input_ids'][idx:idx+1].to(device)
    attention_mask = inputs['attention_mask'][idx:idx+1].to(device,dtype=torch.float16)
    pixel_values = inputs['pixel_values'][idx:idx+1].to(device)

    vision_feature_layer = None
    vision_feature_select_strategy = None
    vision_feature_layer = (vision_feature_layer if vision_feature_layer is not None else model.config.vision_feature_layer)
    vision_feature_select_strategy = (vision_feature_select_strategy
                                        if vision_feature_select_strategy is not None
                                        else model.config.vision_feature_select_strategy)

    with torch.no_grad():
        inputs_embeds = model.get_input_embeddings()(input_ids)

    with torch.no_grad():
        image_features = model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy
            )
        image_features = torch.cat(image_features, dim=0)

    special_image_mask = input_ids == model.config.image_token_id
    n_image_tokens = (special_image_mask).sum()
    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    if inputs_embeds[special_image_mask].numel() != image_features.numel():
        n_image_features = image_features.shape[0] * image_features.shape[1]
        raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    cache_position = None
    past_key_values = None
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position: torch.Tensor = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    position_ids = None
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # create position embeddings to be shared across the decoder layers
    position_embeddings = model.language_model.rotary_emb(inputs_embeds, position_ids)
    with torch.no_grad():
        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=False,
        )
    
    outputs = LRA(outputs['last_hidden_state'].to(dtype=torch.float32)[0],nrank)
    outputs = outputs.reshape(1,outputs.shape[0],-1).to(device, dtype=torch.float16)
    outputs = model.language_model.norm(outputs)

    # Convert to logits with the LM head
    with torch.no_grad():
        logits = model.lm_head(outputs[:, -1:, :])  # shape: (B, 1, vocab_size)

    # Choose next token (greedy here; you can replace with sampling logic)
    next_token_id = torch.argmax(logits, dim=-1)

    # We'll store generated tokens here
    generated = input_ids.clone()
    past_key_values = None

    # Append token to the generated sequence
    generated = torch.cat([generated, next_token_id], dim=-1)
    attention_mask = torch.cat([attention_mask,torch.tensor([[1]],device=device)],dim=-1)

    # Generate exactly <> new tokens
    for step in range(50):
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                input_ids=generated,       # feed only the last token
                # past_key_values=past_key_values,   # speed via caching
                # use_cache=True,                    # ensure caching is enabled
                output_hidden_states=True,         # to get hidden states
                return_dict=True
            )
        
        # outputs.decoder_hidden_states is a tuple: (embeddings, layer1, ..., last_layer)
        decoder_hidden_states = outputs.hidden_states

        decoder_hidden_states = LRA(decoder_hidden_states[-1].to(dtype=torch.float32)[0], nrank)
        decoder_hidden_states = decoder_hidden_states.reshape(1,decoder_hidden_states.shape[0],-1).to(device, dtype=torch.float16)
        decoder_hidden_states = model.language_model.norm(decoder_hidden_states)

        # Extract the hidden state of the last layer, last token
        last_hidden = decoder_hidden_states[:, -1:, :]  # shape: (B, 1, hidden_size)

        # Convert to logits with the LM head
        with torch.no_grad():
            logits = model.lm_head(last_hidden)  # shape: (B, 1, vocab_size)

        # Choose next token (greedy here; you can replace with sampling logic)
        next_token_id = torch.argmax(logits, dim=-1)

        # Append token to the generated sequence
        generated = torch.cat([generated, next_token_id], dim=-1)
        attention_mask = torch.cat([attention_mask,torch.tensor([[1]],device=device)],dim=-1)

        if next_token_id.item() == processor.tokenizer.eos_token_id:
            break

    # Decode the full generated sequence
    decoded = decoded + [processor.tokenizer.decode(generated[0], skip_special_tokens=True)]


# for idx,i in enumerate(predictions[5]):
#     print(f"gold reference is {ds_main[idx]['gold_reference']} and prediction is ", i.replace('\n',''))

# for k,v in predictions.items():
#     y_pred = [1 if 'yes' in i.lower() else 0 for i in v]
#     accuracy = np.sum([1 if i==j else 0 for i,j in zip(y_pred, gold_reference)])/len(gold_reference)
#     print(k, accuracy)

