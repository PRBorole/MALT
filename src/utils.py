import json
import six
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModel
from transformers import Gemma3ForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
from transformers import AutoModelForImageTextToText
import torch
import re
from datasets import load_dataset, concatenate_datasets

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

def load_model_and_processor(
    model_name='llava-1.5-7b', 
    model_path=None, 
    low_cpu_mem_usage=True, 
    attn_implementation='sdpa', 
    torch_dtype='auto'
):
    if model_name == 'llava-1.5-7b':
        model = AutoModelForImageTextToText.from_pretrained(
            f"{model_path}/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation
        )
        processor = AutoProcessor.from_pretrained(
            f"{model_path}/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            local_files_only=True,
            padding_side='left'
        )

    elif model_name == 'llava-v1.6-mistral-7b':
        model = LlavaNextForConditionalGeneration.from_pretrained(
            f"{model_path}/models--llava-hf--llava-v1.6-mistral-7b-hf/snapshots/52320fb52229c8d942b1dcb8b63b3dc8087bc83b",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation
        )
        processor = LlavaNextProcessor.from_pretrained(
            f"{model_path}/models--llava-hf--llava-v1.6-mistral-7b-hf/snapshots/52320fb52229c8d942b1dcb8b63b3dc8087bc83b",
            local_files_only=True,
            padding_side='left'
        )
    
    elif model_name == 'llava-v1.6-vicuna-7b':
        model = LlavaNextForConditionalGeneration.from_pretrained(
            f"{model_path}/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation,
        )

        processor = AutoProcessor.from_pretrained(
            f"{model_path}/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23",
            local_files_only=True,
            padding_side='left'
        )

    elif model_name == 'Qwen2.5-VL-7B':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            f"{model_path}/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            use_safetensors=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation
        )
        processor = AutoProcessor.from_pretrained(
            f"{model_path}/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
            local_files_only=True,
            padding_side='left'
        )

    elif model_name == 'Qwen2.5-VL-3B':
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            f"{model_path}/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            use_safetensors=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation
        )
        processor = AutoProcessor.from_pretrained(
            f"{model_path}/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3",
            local_files_only=True,
            padding_side='left'
        )

    elif model_name == 'Qwen3-VL-4B-Instruct':
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            f"{model_path}/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17/",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            use_safetensors=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation
        )
        processor = AutoProcessor.from_pretrained(
            f"{model_path}/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17/",
            local_files_only=True,
            padding_side='left'
        )

    elif model_name == 'Qwen3-VL-4B-Thinking':
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            f"{model_path}/models--Qwen--Qwen3-VL-4B-Thinking/snapshots/1de27d8c51f12e819435303b9e84c4e25ba8401e/",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            use_safetensors=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation
        )
        processor = AutoProcessor.from_pretrained(
            f"{model_path}/models--Qwen--Qwen3-VL-4B-Thinking/snapshots/1de27d8c51f12e819435303b9e84c4e25ba8401e/",
            local_files_only=True,
            padding_side='left'
        )

    elif model_name == 'Qwen3-VL-8B-Instruct':
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            f"{model_path}/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            use_safetensors=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation
        )
        processor = AutoProcessor.from_pretrained(
            f"{model_path}/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/",
            local_files_only=True,
            padding_side='left'
        )
        
    elif model_name == 'Qwen3-VL-8B-Thinking':
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            f"{model_path}/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/41ea130ce6eaaf7829c72dfc0e4597d49741ed18/",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            use_safetensors=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation
        )
        processor = AutoProcessor.from_pretrained(
            f"{model_path}/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/41ea130ce6eaaf7829c72dfc0e4597d49741ed18/",
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
            f"{model_path}/models--allenai--Molmo-7B-D-0924/snapshots/546563490f0318d3a7b9c1b1e0f73927f6eb5c2e",
            local_files_only=True,
            torch_dtype=torch_dtype, 
            use_safetensors=True,
            trust_remote_code=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation
        ).eval()
        
        processor = AutoProcessor.from_pretrained(
            f"{model_path}//models--allenai--Molmo-7B-D-0924/snapshots/546563490f0318d3a7b9c1b1e0f73927f6eb5c2e",
            local_files_only=True,
            trust_remote_code=True,
        )

    elif model_name=='deepseek-ocr':
        model = AutoModel.from_pretrained(
            f"{model_path}//models--deepseek-ai--DeepSeek-OCR/snapshots/9f30c71f441d010e5429c532364a86705536c53a",
            local_files_only=True,
            torch_dtype=torch.bfloat16, 
            use_safetensors=True,
            trust_remote_code=True,
            cache_dir=f"{model_path}/.cache/",
            # low_cpu_mem_usage=True,
            attn_implementation='flash_attention_2').eval()
        
        processor = AutoProcessor.from_pretrained(
            f"{model_path}//models--deepseek-ai--DeepSeek-OCR/snapshots/9f30c71f441d010e5429c532364a86705536c53a",
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

def build_prompt(
    ds_item: str,
    ds_name: str, 
    hint_use: bool=True
)-> str:
    """
    Build prompt
    """
    question = ds_item["question"]
    hint = ""
    if ds_name=='ScienceQA':
        hint = f"HINT: {ds_item['hint']}\n" if hint_use else ""
    
    choices_str = ""
    if 'choices' in ds_item:
        choices = ds_item["choices"] 
        nchoices = len(choices)-1
        choices_str = "\n".join(
            [f"{i}: {c}" for i, c in enumerate(choices)]
        )
        choices_str = (
            f"OPTIONS:\n{choices_str}\n\n"
            f"Answer with the correct option from 0 to {len(choices)-1}. Only provide the numeric and nothing else \n\n"
        )
    else:
        choices_str = "Only provide answer and nothing else \n\n"
    
    prompt = (
        f"QUESTION: {question}\n"
        f"{hint}"
        f"{choices_str}\n"
        f"Answer: "
    )
    return prompt


def process_input_data(
    ds, 
    processor, 
    image_dir_path=None, 
    image_use=True,
    device="cuda", 
    model_name='llava-1.5-7b'
):
    """
    Process the input data by loading images and prompts from the dataset.
    
    arguments:
    ds: list of dictionaries, 
    image_path: str, path to the directory containing images.
    image_use: bool, whether to include image in the prompt
    processor: processor for preparing inputs for the model.
    device: str, device to run the model on (default is 'cuda').
    model_name: str, type of model being used (default is 'llava-1.5-7b').
    
    returns:
    images: list of PIL Image objects, loaded from the specified image paths.
    prompts: list of strings, prompts corresponding to each image.
    inputs: list of dictionaries, processed inputs ready for the model.
    """
    
    image_path = None
    if image_dir_path is not None:
        image_path = [image_dir_path + ds_item['image'] for ds_item in ds]
        images = process_image(image_path)
    else:
        images = [ds_item['image'] for ds_item in ds]
    prompts = [ds_item['prompt'] for ds_item in ds]
    
    if model_name in [
        'llava-1.5-7b',
        'llava-v1.6-mistral-7b', 
        'llava-v1.6-vicuna-7b',
        'Qwen2.5-VL-7B', 
        'Qwen2.5-VL-3B', 
        'Qwen3-VL-4B-Instruct', 
        'Qwen3-VL-4B-Thinking', 
        'Qwen3-VL-8B-Instruct', 
        'Qwen3-VL-8B-Thinking'
    ]:
        texts = []
        for idx, prompt in enumerate(prompts):
            text = [
                    {
                        "role": "user",
                        "content": [
                            { "type": "image", "url": f"{image_path[idx]}"}
                                if image_path is not None
                                else { "type": "image", "image": images[idx]},
                            { "type": "text", "text": f" {prompt}" }
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
        if not image_use:
            images = None
        inputs = processor.process(
            images=images if type(images) is not list else images[0], # Question: does it only take single image?
            text=texts[0], 
            padding=True, 
            return_tensors="pt"
        )
        inputs = {k: v.to(device).unsqueeze(0) if hasattr(v, "to") else v for k, v in inputs.items()}
        return images, texts, inputs
    
    else:
        raise ValueError("Unsupported model type")
    
    if not image_use:
            images = None
    
    inputs = processor(images=images, text=texts, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    
    return images, texts, inputs

def batch_inference(
    ds, 
    model, 
    processor, 
    image_dir_path=None, 
    image_use=True,
    device="cuda", 
    batch_size=8,
    max_new_tokens=50, 
    model_name='llava-1.5-7b'
):
    """
    Perform batch inference on the dataset using the specified model and processor.
    
    arguments:
    ds: list of dictionaries, dataset containing image paths and prompts.
    image_dir_path: str, path to the directory containing images.
    image_use: bool, whether to include image in the prompt.
    model: pre-trained model for inference.
    processor: processor for preparing inputs for the model.
    device: str, device to run the model on (default is 'cuda').
    batch_size: int, number of samples to process in each batch (default is 8)
    max_new_tokens: int, maximum number of new tokens to generate (default is 50).
    model_name: str, type of model being used (default is 'llava-1.5-7b').
    
    returns:
    all_model_answers: list of strings, model-generated answers for each input in the dataset.
    """
    
    all_model_answers = []
    num_samples = len(ds)
    batch_size = batch_size if model_name !='molmoD-7B' else 1
    
    for start_idx in tqdm(range(0, len(ds), batch_size)):
        end_idx = min(start_idx + batch_size,num_samples)
        batch = ds[start_idx:end_idx] if isinstance(ds, list) else ds.select(range(start_idx, end_idx))
        
        try:
            _, _, inputs = process_input_data(
                ds=batch, 
                image_dir_path=image_dir_path,
                image_use=image_use,
                processor=processor,
                device=device,
                model_name=model_name
            )
            
            with torch.no_grad():
                if 'molmo' in model_name:
                    outputs = model.generate_from_batch(inputs, 
                            GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
                            tokenizer=processor.tokenizer,
                            use_cache=False)
                    model_answers = processor.tokenizer.decode(outputs[0,inputs['input_ids'].size(1):], skip_special_tokens=True)
                    if type(model_answers)!=list:
                        model_answers = [model_answers]
                else:
                    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
                    model_answers = processor.batch_decode(outputs, skip_special_tokens=True)
                outputs = outputs.detach().cpu()
                
            all_model_answers = all_model_answers + model_answers
        
        except torch.OutOfMemoryError:
            print(f"Skipping sample {start_idx}")
            torch.cuda.empty_cache()
            continue
    
    return all_model_answers

def process_predictions(model_name, all_model_outputs):
    """
    Perform processing of prediction outputs from various VLMs.
    
    arguments:
    model_name: pre-trained model for inference.
    all_model_outputs: list of model outputs.

    returns:
    predictions: list of strings, answer for the question.
    """
    
    if model_name == 'llava-1.5-7b':
        predictions = [
            i.split('ASSISTANT:')[-1].lstrip().rstrip()
            if len(i.split('ASSISTANT:'))>1
            else '10000' #random large number for non split
            for i in all_model_outputs
        ]
    
    elif model_name == 'llava-v1.6-mistral-7b':
        if 'answer' not in all_model_outputs[0].split('[/INST] ')[1].lstrip():
            predictions = [
                i.split('[/INST] ')[1].lstrip().rstrip() 
                if len(i.split('[/INST] '))>1
                else '10000' #random large number for non split
                for i in all_model_outputs
            ]
        else: # This for MATE dataset specifically
            predictions = [i.split('[/INST]')[1].lstrip().split('"answer":')[1].lstrip().replace('}','').replace(')','').replace('"','').rstrip()
                       for i in all_model_outputs]
        
    elif model_name=='llava-v1.6-vicuna-7b':
        if 'answer' not in all_model_outputs[0].split('ASSISTANT:')[1].lstrip():
            predictions = [
                i.split('ASSISTANT:')[1].lstrip().rstrip() 
                if len(i.split('ASSISTANT:'))>1
                else '10000' #random large number for non split
                for i in all_model_outputs
            ]
        else: # This for MATE dataset specifically
            predictions = [i.split('ASSISTANT: ASSISTANT:')[1].lstrip().split('"answer":')[1].lstrip().replace('}','').replace(')','').replace('"','').rstrip()
                          for i in all_model_outputs]
    
    elif 'Qwen' in model_name: 
        predictions = []
        for idx, i in enumerate(all_model_outputs):
            if type(i)==list:
                assert len(i)==1, "all_model_output is list of list with each list containing multiple answers"
                i = i[0]
            if 'answer' not in i.split('\nassistant\n')[1].lstrip():
                predictions = predictions + [i.split('\nassistant\n')[1].lstrip()]
            else: # This for MATE dataset specifically
                predictions = predictions + [i.split('\nassistant\n')[1].lstrip().split('"answer":')[1].lstrip().replace('}','').replace('"','').replace(')','')]
    
    elif model_name=='gemma3-4B' or model_name=='gemma3-12B':
        predictions = [i.split('ASSISTANT:\nmodel\n')[1].lstrip().split('"answer":')[1].lstrip().replace('}','').replace('"','').replace(')','')
                       for i in all_model_outputs]
        
    elif model_name=='molmoD-7B':
        predictions  = [i.lstrip() if '"' not in i else i.split('"')[1]
                        for i in all_model_outputs]
        
    return predictions


def load_dataset_formatted(
    ds_name: str
):
    """
    Helper function to load dataset and format consistently
    Return:
        ds: Formatted dataset
    """
    if ds_name=='MATE':
        # ds = load_dataset("HiTZ/MATE", split='cross_modal')
        # ds = ds.rename_column("answer","gold_reference")
        # ds = ds.rename_column("task", "tag")
        pass
    elif ds_name=='ScienceQA':
        ds = load_dataset("derek-thomas/ScienceQA", split="test")
    elif ds_name=='MicroVQA':
        ds = load_dataset("jmhb/microvqa", split="test")
        ds = ds.rename_column("images_list", "image")
        ds = ds.rename_column("correct_index","gold_reference")
        ds = ds.rename_column("task_str", "tag")
    elif ds_name=='MathVision':
        ds = load_dataset("MathLLMs/MathVision",split='test')
        ds = ds.remove_columns(["image"])
        ds = ds.rename_column("decoded_image","image")
        ds = ds.rename_column("options","choices")
        ds = ds.rename_column("level", "tag")
        ds = ds.filter(lambda x: len(x['choices'])>1)
        mapper_dict = {i:idx for idx, i in enumerate(sorted(set(ds[:]['answer'])))} # alpha->ints for consistency
        ds = ds.add_column("gold_reference", [mapper_dict[i] for i in ds[:]['answer']])
    elif ds_name=='HallusionBench':
        ds = load_dataset("lmms-lab/HallusionBench",split='image')
        ds = ds.add_column("choices", [[0,1] for i in range(len(ds))])
        ds = ds.add_column("gold_reference", [int(i) for i in ds[:]['gt_answer']])
        ds = ds.filter(lambda x: x['visual_input']=='1') # 1 is answer dependent on image
        ds = ds.filter(lambda x: x['subcategory']!='video') # remove video question
        ds = ds.rename_column("subcategory", "tag")
    elif ds_name=='BLINK':
        ls = [
            "Counting",
            "IQ_Test",
            "Object_Localization",
            "Relative_Depth",
            "Relative_Reflectance",
            "Spatial_Relation",
        ]
        parts = []
        for subset in ls:
            ds_ = load_dataset("BLINK-Benchmark/BLINK", subset, split="val")
            ds_ = ds_.add_column("subset", [subset] * len(ds_))   # keep provenance
            parts.append(ds_)
        # Combine into one big dataset
        ds = concatenate_datasets(parts)
        mapper_dict = {'A':0, 'B':1, 'C':2, 'D':3}
        ds = ds.add_column("gold_reference", [mapper_dict[re.sub(r'[^a-zA-Z0-9]', '', i)] for i in ds[:]['answer']])
        ds = ds.rename_column("sub_task", "tag")
        ds = ds.rename_column("image_1", "image")
    elif ds_name=='VstarBench':
        ds = load_dataset("lmms-lab/vstar-bench", split='test')
        ds = ds.rename_column("category", "tag")
        mapper_dict = {'A':0, 'B':1, 'C':2, 'D':3}
        ds = ds.add_column("gold_reference", [mapper_dict[i] for i in ds[:]['label']])
        ds = ds.map(lambda x: {"question": x["text"].replace("(A)", "0").replace("(B)", "1").replace("(C)", "2").replace("(D)", "3")})
    else:
        raise ValueError("Unsupported dataset")
    return ds

# [i['image'].size for i in ds.select([128,129,130])]
# [i['image'].size[0]*i['image'].size[1] for i in ds.select([128,129,130])]