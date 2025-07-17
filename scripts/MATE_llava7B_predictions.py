from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import LlavaForConditionalGeneration, AutoModelForImageTextToText
import torch
from src.utils import *
import pandas as pd

def main(image_dir_path=None, ds_path=None):
    """
    Main function to run the llava model inference on MATE.
    It loads the dataset, processes the input data, and performs inference using the model.
    """
    # image_dir_path = './../datasets/MATE-dev/img/'
    # ds = load_jsonl_file('./../datasets/MATE-dev/mm_0shot_llava_hfllava_1.5_7b_hf.jsonl')
    ds = load_jsonl_file(ds_path)

    model, processor, device = load_model_and_processor(model_name='llava_1.5_7b')

    idx = len(ds)
    all_model_outputs = batch_inference(
        ds[:idx], 
        image_dir_path, 
        model, processor, 
        device=device, 
        batch_size=8, 
        max_new_tokens=50, 
        model_name='llava_1.5_7b'
    )

    predictions = [i.split('ASSISTANT:')[1].lstrip() for i in all_model_outputs]
    original_pred = [i['prediction'] for i in ds[:idx]]

    prediction_df = pd.DataFrame({'image':[ds[i]['image'] for i in range(len(ds))],
                                'output':all_model_outputs,
                                'prediction': predictions,
                                'paper_prediction': original_pred,
                                'gold_reference': pd.DataFrame(ds)['gold_reference'],
                                'object_count': [data['object_count'] for data in ds]
                                })


    prediction_df['prediction_lower'] = prediction_df['prediction'].apply(lambda x: x.lower())
    prediction_df['gold_reference_lower'] = prediction_df['gold_reference'].apply(lambda x: x.lower())
    prediction_df['paper_prediction_lower'] = prediction_df['paper_prediction'].apply(lambda x: x.lower())


    print("mismatch current vs MATE paper prediction: ",
        len(prediction_df[prediction_df['prediction_lower']!=prediction_df['paper_prediction_lower']]))
    print("mismatch current vs gold reference: ",
        len(prediction_df[prediction_df['prediction_lower']!=prediction_df['gold_reference_lower']]))
    print("mismatch MATE paper prediction vs gold reference: ",
        len(prediction_df[prediction_df['paper_prediction_lower']!=prediction_df['gold_reference_lower']]))

    prediction_df.to_csv('./results/llava7B/MATE_prediction.csv')
    # prediction_df = pd.read_csv('./results/llava7B/MATE_prediction.csv', index_col=0)

