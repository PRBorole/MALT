import json
import six
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
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

def load_model_and_processor(model_name='llava_1.5_7b'):
    if model_name == 'llava_1.5_7b':
        model = AutoModelForImageTextToText.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/63f0593ce09615bbc4a43a4c13143ba3c389e53a",
            local_files_only=True,
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        processor = AutoProcessor.from_pretrained(
            "/home/jovyan/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/63f0593ce09615bbc4a43a4c13143ba3c389e53a",
            local_files_only=True
        )
    else:
        raise ValueError("Unsupported model type")
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
        prompts = [f"<image>\n {prompt}" for prompt in prompts]
    else:
        raise ValueError("Unsupported model type")
    
    inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    
    return images, prompts, inputs

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

        # prompts = [f"<image>\nContext: {context}\nQuestion: {question}\nAnswer:" for context, question in zip(contexts, questions)]

        with torch.no_grad():
            if model_name == 'llava_1.5_7b':
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
                model_answers = processor.batch_decode(outputs, skip_special_tokens=True)
            else:
                raise ValueError("Unsupported model type")
            
        all_model_answers.extend(model_answers)

    return all_model_answers