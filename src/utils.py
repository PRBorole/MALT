import json
import six
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import Gemma3ForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from transformers import LlavaForConditionalGeneration, AutoModelForImageTextToText
import torch

def get_gpu_memory():
    free, total = torch.cuda.mem_get_info()
    return print(f"free {free/1024**2} MB, total {total/1024**2} MB")

def load_jsonl_file(jsonl_file_path):
    # './../datasets/MATE-dev/mm_0shot_llava_hfllava_1.5_7b_hf.jsonl'
    # Adapted from https://github.com/hitz-zentroa/MATE/blob/main/src/datasource/utils.py
    """
    Load a JSONL file and return its content as a list of dictionaries.
    Each line in the file is expected to be a valid JSON object.
    
    arguments:
    jsonl_file_path: str, path to the JSONL file to be loaded.

    returns:
    result: list of dictionaries, where each dictionary corresponds to a line in the JSONL file
    """
    result = []
    jsonl_file_path
    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = six.ensure_text(line, "utf-8")
            example = json.loads(line)
            result.append(example)
    return result

def load_model_and_processor(model_name='llava_1.5_7b', low_cpu_mem_usage=True, attn_implementation='sdpa', torch_dtype='auto'):
    if model_name == 'llava_1.5_7b':
        model = AutoModelForImageTextToText.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/63f0593ce09615bbc4a43a4c13143ba3c389e53a",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation
        )
        processor = AutoProcessor.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/63f0593ce09615bbc4a43a4c13143ba3c389e53a",
            local_files_only=True,
            padding_side='left'
        )

    elif model_name == 'llava-v1.6-mistral-7b':
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--llava-hf--llava-v1.6-mistral-7b-hf/snapshots/740747880a3f38134c6322f0115687dcb7eafa20",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation
        )
        processor = LlavaNextProcessor.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--llava-hf--llava-v1.6-mistral-7b-hf/snapshots/740747880a3f38134c6322f0115687dcb7eafa20",
            local_files_only=True,
            padding_side='left'
        )
    
    elif model_name == 'llava-v1.6-vicuna-7b':
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/ffc6ee0d757164e29a7d0ccbaafb93f3085105d6",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation,
        )

        processor = LlavaNextProcessor.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/ffc6ee0d757164e29a7d0ccbaafb93f3085105d6",
            local_files_only=True,
            padding_side='left'
        )

    elif model_name == 'Qwen2.5-VL-7B':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/e444c4f900cc3c7b3fb49dba54d1b4634f958c73",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            use_safetensors=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation
        )
        processor = AutoProcessor.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/e444c4f900cc3c7b3fb49dba54d1b4634f958c73",
            local_files_only=True,
            padding_side='left'
        )

    elif model_name == 'Qwen2.5-VL-3B':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/5ec6d69a567621b310d3bc0ec334e79430d137e2",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            use_safetensors=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation
        )
        processor = AutoProcessor.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/5ec6d69a567621b310d3bc0ec334e79430d137e2",
            local_files_only=True,
            padding_side='left'
        )

    elif model_name == 'gemma3-4B':
        model = Gemma3ForConditionalGeneration.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/fb45cab2e2d05204ac7e12f3051d144981d59f41",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            use_safetensors=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation
        ).eval()
        processor = AutoProcessor.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/fb45cab2e2d05204ac7e12f3051d144981d59f41",
            local_files_only=True
        )
    
    elif model_name == 'gemma3-12B':
        model = Gemma3ForConditionalGeneration.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/28bf32ec0b699658135870a6c18cb6aa7d050294",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            use_safetensors=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation
        ).eval()
        processor = AutoProcessor.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/28bf32ec0b699658135870a6c18cb6aa7d050294",
            local_files_only=True
        )

    elif model_name == 'molmoD-7B':
        model = AutoModelForCausalLM.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--allenai--Molmo-7B-D-0924/snapshots/cf359d3f08781b4ca5ea34fe058e812efd216d1a",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            use_safetensors=True,
            trust_remote_code=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation
        ).eval()
        processor = AutoProcessor.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--allenai--Molmo-7B-D-0924/snapshots/cf359d3f08781b4ca5ea34fe058e812efd216d1a",
            local_files_only=True,
            trust_remote_code=True,
        )

    else:
        raise ValueError("Unsupported model type")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, processor, device

def process_image(image_path):
    """
    Process an image by opening it and converting it to RGB format.
    
    arguments:
    image_path: str, path to the image file.

    returns:
    image: list of PIL Image objects the processed image in RGB format.
    """
    
    assert(image_path is not None), "Image path cannot be None"
    images = [Image.open(path).convert("RGB") for path in image_path]
    return images


def process_input_data(ds, image_dir_path, processor, device="cuda", model_name='llava_1.5_7b'):
    """
    Process the input data by loading images and prompts from the dataset.
    
    arguments:
    ds: list of dictionaries, dataset containing image paths and prompts.
    image_path: str, path to the directory containing images.
    processor: processor for preparing inputs for the model.
    device: str, device to run the model on (default is 'cuda').
    model_name: str, type of model being used (default is 'llava_1.5_7b').

    returns:
    images: list of PIL Image objects, loaded from the specified image paths.
    prompts: list of strings, prompts corresponding to each image.
    inputs: list of dictionaries, processed inputs ready for the model.
    """
    image_path = [image_dir_path + data['image'] for data in ds]
    images = process_image(image_path)
    prompts = [data['prompt'] for data in ds]

    if model_name == 'llava_1.5_7b':
        texts = [f"<image>\n {prompt}" for prompt in prompts]

    elif model_name == 'llava-v1.6-mistral-7b' or model_name == 'llava-v1.6-vicuna-7b':
        texts = []
        for idx, prompt in enumerate(prompts):
            text = [
                            {
                                "role": "user",
                                "content": [
                                    { "type": "image", "url": f"{image_path[idx]}"},
                                    { "type": "text", "text": f" {prompt}" }
                                ]
                            }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                                    text, tokenize=False, add_generation_prompt=True
            )
            texts = texts + [text]

    elif model_name=='Qwen2.5-VL-7B' or model_name == 'Qwen2.5-VL-3B':
        texts = []
        for idx, prompt in enumerate(prompts):
            text = [
                            {
                                "role": "user",
                                "content": [
                                    { "type": "image", "url": f"{image_path[idx]}" },
                                    { "type": "text", "text": f"{prompt}" }
                                ]
                            }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                                    text, tokenize=False, add_generation_prompt=True
            )
            texts = texts + [text]

    elif model_name=='gemma3-4B' or model_name=='gemma3-12B':
        texts = []
        for idx, prompt in enumerate(prompts):
            text = [
                            {
                                "role": "user",
                                "content": [
                                    { "type": "image", "image": f"{image_path[idx]}"},
                                    { "type": "text", "text": f"{prompt}" }
                                ]
                            }
            ]
            # Preparation for inference
            text = processor.apply_chat_template(
                                    text, tokenize=False, add_generation_prompt=True
            )
            texts = texts + [text]
        images = [[i] for i in images]

    elif model_name=='molmoD-7B':
        texts = prompts
        inputs = processor.process(images=images, text=texts[0], padding=True, return_tensors="pt")
        inputs = {k: v.to(device).unsqueeze(0) if hasattr(v, "to") else v for k, v in inputs.items()}
        return images, texts, inputs

    else:
        raise ValueError("Unsupported model type")
    
    inputs = processor(images=images, text=texts, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    
    return images, texts, inputs

def batch_inference(ds, image_dir_path, model, processor, device="cuda", batch_size=8, max_new_tokens=50, model_name='llava_1.5_7b'):
    """
    Perform batch inference on the dataset using the specified model and processor.
    
    arguments:
    ds: list of dictionaries, dataset containing image paths and prompts.
    image_dir_path: str, path to the directory containing images.
    model: pre-trained model for inference.
    processor: processor for preparing inputs for the model.
    device: str, device to run the model on (default is 'cuda').
    batch_size: int, number of samples to process in each batch (default is 8)
    max_new_tokens: int, maximum number of new tokens to generate (default is 50).
    model_name: str, type of model being used (default is 'llava_1.5_7b').

    returns:
    all_model_answers: list of strings, model-generated answers for each input in the dataset.
    """
    
    all_model_answers = []
    num_samples = len(ds)
    for start_idx in tqdm(range(0, len(ds), batch_size)):
        end_idx = min(start_idx + batch_size,num_samples)
        _, _, inputs = process_input_data(ds=ds[start_idx:end_idx], 
                                             image_dir_path=image_dir_path,
                                             processor=processor,
                                             device=device,
                                             model_name=model_name)

        with torch.no_grad():
            if 'molmo' in model_name:
                outputs = model.generate_from_batch(inputs, GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
                                                tokenizer=processor.tokenizer)
                model_answers = processor.tokenizer.decode(outputs[0,inputs['input_ids'].size(1):], skip_special_tokens=True)
            else:
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
                model_answers = processor.batch_decode(outputs, skip_special_tokens=True)
            outputs = outputs.detach().cpu()
            
        all_model_answers = all_model_answers + [model_answers]

    return all_model_answers