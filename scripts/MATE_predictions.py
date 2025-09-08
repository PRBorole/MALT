from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import LlavaForConditionalGeneration, AutoModelForImageTextToText
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from src.utils import *
import pandas as pd
import torch._dynamo

model_name = 'molmoD-7B' # molmoD-7B, llava_1.5_7b, Qwen2.5-VL-3B, Qwen2.5-VL-7B,  llava-v1.6-mistral-7b,  llava-v1.6-vicuna-7b, gemma3-4B, gemma3-12B

if model_name=='gemma3-4B' or model_name=='gemma3-12B':
    torch._dynamo.config.cache_size_limit = 32

# def main(model_name, image_dir_path=None, ds_path=None):
"""
Main function to run the llava model inference on MATE.
It loads the dataset, processes the input data, and performs inference using the model.
"""
image_dir_path = './../datasets/MATE-dev/img/'
ds = load_jsonl_file('./../datasets/MATE-dev/mm_0shot_llava_hfllava_1.5_7b_hf.jsonl')
mate_df = pd.read_csv('./data/mate_df.csv',index_col=0)

model, processor, device = load_model_and_processor(model_name=model_name, attn_implementation='eager')

all_model_outputs = batch_inference(
    ds, 
    image_dir_path, 
    model, processor, 
    device=device, 
    batch_size=1, 
    max_new_tokens=50, 
    model_name=model_name
)

if model_name == 'llava_1.5_7b':
    predictions = [i.split('ASSISTANT:')[1].lstrip() for i in all_model_outputs]

elif model_name == 'llava-v1.6-mistral-7b':
    predictions = [i.split('[/INST]')[1].lstrip().split('"answer":')[1].lstrip().replace('}','').replace(')','').replace('"','').rstrip()
                   for i in all_model_outputs]
    
elif model_name=='llava-v1.6-vicuna-7b':
    predictions = [i.split('ASSISTANT: ASSISTANT: ')[1].lstrip().split('"answer":')[1].lstrip().replace('}','').replace(')','').replace('"','').rstrip()
                   for i in all_model_outputs]

elif model_name=='Qwen2.5-VL-7B' or model_name=='Qwen2.5-VL-3B':
    predictions = []
    for idx, i in enumerate(all_model_outputs):
        if 'answer' not in i.split('ASSISTANT:\nassistant\n')[1].lstrip():
            predictions = predictions + [i.split('ASSISTANT:\nassistant\n')[1].lstrip()]
        else:
            predictions = predictions + [i.split('ASSISTANT:\nassistant\n')[1].lstrip().split('"answer":')[1].lstrip().replace('}','').replace('"','').replace(')','')]

elif model_name=='gemma3-4B' or model_name=='gemma3-12B':
    predictions = [i.split('ASSISTANT:\nmodel\n')[1].lstrip().split('"answer":')[1].lstrip().replace('}','').replace('"','').replace(')','')
                   for i in all_model_outputs]
    
elif model_name=='molmoD-7B':
    predictions  = [i.lstrip() if '"' not in i else i.split('"')[1]
                    for i in all_model_outputs]



prediction_df = pd.DataFrame({'image':[ds[i]['image'] for i in range(len(ds))],
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

prediction_df[prediction_df['prediction_lower']!=prediction_df['gold_reference_formatted']][['prediction_lower','gold_reference_formatted']]

if model_name == 'llava_1.5_7b':
    original_pred = [i['prediction'] for i in ds]
    prediction_df['paper_prediction'] = original_pred
    prediction_df['paper_prediction_lower'] = prediction_df['paper_prediction'].apply(lambda x: x.lower())

    print("mismatch current vs MATE paper prediction: ",
        len(prediction_df[prediction_df['prediction_lower']!=prediction_df['paper_prediction_lower']]))

    print("mismatch MATE paper prediction vs gold reference: ",
        len(prediction_df[prediction_df['paper_prediction_lower']!=prediction_df['gold_reference_lower']]))

prediction_df.to_csv(f'./results/molmoB-7B/{model_name}_MATE_prediction.csv')
# prediction_df = pd.read_csv('./results/llava7B/MATE_prediction.csv', index_col=0)

