from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import LlavaForConditionalGeneration, AutoModelForImageTextToText
import torch
from tqdm import tqdm
from PIL import Image
from src.utils import *
import pandas as pd
from torchinfo import summary



image_dir_path = './../datasets/MATE-dev/img/'
ds = load_jsonl_file('./../datasets/MATE-dev/mm_0shot_llava_hfllava_1.5_7b_hf.jsonl')


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

start_idx = 0
end_idx = 1
_, _, inputs = process_input_data(ds=ds[start_idx:end_idx], 
                                    image_dir_path=image_dir_path,
                                    processor=processor,
                                    device=device,
                                    model_name='llava_1.5_7b')

inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])

# Forward pass with output_hidden_states=True
with torch.no_grad():
    outputs = model.language_model(
        attention_mask=inputs['attention_mask'],
        inputs_embeds=inputs_embeds,
        output_hidden_states=True,
        return_dict=True,
    )

# Extract hidden states from all layers
# outputs.hidden_states is a tuple: (embedding_output, layer1_output, ..., last_layer_output)
all_layer_embeddings = outputs.hidden_states  # tuple of tensors, each [batch, seq_len, hidden_dim]
all_layer_embeddings = torch.vstack(all_layer_embeddings) 
image_hidden_layers = all_layer_embeddings

all_layer_embeddings[:,torch.where(inputs['input_ids'] == model.config.image_token_id)[1].tolist(),:].shape


