from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import LlavaForConditionalGeneration, AutoModelForImageTextToText
import torch
from tqdm import tqdm
from PIL import Image
from .utils import *
import pandas as pd
import numpy as np
from torchinfo import summary

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score


class LinearProbe():
    """
    Class to handle probing experiments.
    """
    def __init__(self, model, nobjects, task, mode, target, layers='all'):
        self.nobjects = nobjects
        self.task = task
        self.target = target
        self.model = model
        self.mode = mode

    def get_prompt(self, ds):
        """        
        Get the prompt for the probing experiment.

        arguments:
        ds: dataset entry containing the scene information

        returns:
        prompt: string containing the prompt for the probing task
        """
        if self.mode == 'image':
            prompt = "USER: \nThis image shows a minimalist arrangement of 3D geometric shapes made of different materials and colors." +\
                        "The JSON provided contains information about all objects in the scene (material: metal = shinny, rubber = matte))."+\
                        "\n\nUnderstanding the Coordinate System X, Y, Z:"+\
                        "\n• X (Depth): Represents the depth relative to the camera. Smaller values indicate objects that are farther away."+\
                        "\n• Y (Horizontal Position): Represents the left-to-right position. A value of zero means the object is centered in the scene, negative values place the object to the left, and positive values to the right."+\
                        "\n• Z (Vertical Position): Represents the height of the object\'s center point. Larger values correspond to higher vertical positions."+\
                        "\n\nHere is the JSON containing details about all objects in the scene in the image:"+\
                        f"\n\n {({'camera_location': ds['scene']['camera_location'], 'objects': [{key: data[key] for key in data.keys() if key!=self.task} for data in ds['scene']['objects']]})}"+\
                        f"\n Now only answer YES or NO, is an object of {self.task} {self.target} present in the image? ASSISTANT: "
        elif self.mode == 'text':
            prompt = "USER: \nThis image shows a minimalist arrangement of 3D geometric shapes made of different materials and colors." +\
                        "The JSON provided contains information about all objects in the scene."+\
                        "\n\nUnderstanding the Coordinate System X, Y, Z:"+\
                        "\n• X (Depth): Represents the depth relative to the camera. Smaller values indicate objects that are farther away."+\
                        "\n• Y (Horizontal Position): Represents the left-to-right position. A value of zero means the object is centered in the scene, negative values place the object to the left, and positive values to the right."+\
                        "\n• Z (Vertical Position): Represents the height of the object\'s center point. Larger values correspond to higher vertical positions."+\
                        "\n\nHere is the JSON containing details about all objects in the scene in the image:"+\
                        f"\n\n {({'camera_location': ds['scene']['camera_location'], 'objects': [ds['scene']['objects']]})}"+\
                        f"\n Now only answer YES or NO, is an object of {self.task} {self.target} present in the image? ASSISTANT: "
        return prompt

    def get_layer_llava_embeddings(self, pixel_values, input_ids, attention_mask, mean_dim=1):
        """
        Get language model per layer embeddings from the llava model.

        arguments:
        pixel_values: tensor of shape (batch_size, channels, height, width) containing the pixel values of the images
        input_ids: tensor of shape (batch_size, sequence_length)
        attention_mask: tensor of shape (batch_size, sequence_length) indicating which tokens are padded
        mean_dim: dimension along which to take the mean of the embeddings, default is 1 for sententce-level embeddings

        returns:
        all_layer_embeddings: numpy array of shape (batch_size, n_layers, embedding_size) containing the embeddings for each layer
        """

        vision_feature_layer = None
        vision_feature_select_strategy = None
        vision_feature_layer = (vision_feature_layer if vision_feature_layer is not None else self.model.config.vision_feature_layer)
        vision_feature_select_strategy = (vision_feature_select_strategy
                                            if vision_feature_select_strategy is not None
                                            else self.model.config.vision_feature_select_strategy)

        pixel_values = pixel_values
        input_ids = input_ids
        attention_mask = attention_mask

        with torch.no_grad():
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=pixel_values,
                                                            vision_feature_layer=vision_feature_layer,
                                                            vision_feature_select_strategy=vision_feature_select_strategy)
            image_features = torch.cat(image_features, dim=0)

        special_image_mask = input_ids == self.model.config.image_token_id
        n_image_tokens = (special_image_mask).sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            n_image_features = image_features.shape[0] * image_features.shape[1]
            raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Forward pass with output_hidden_states=True
        all_layer_embeddings = []
        with torch.no_grad():
            outputs = self.model.language_model(attention_mask=attention_mask,
                                                inputs_embeds=inputs_embeds,
                                                output_hidden_states=True,
                                                padding=True,
                                                return_dict=True)

            # Extract hidden states from all layers
            hidden_states = torch.vstack(outputs.hidden_states).cpu().detach().numpy()
            image_tokens = (input_ids==self.model.config.image_token_id).cpu().detach().numpy().reshape(-1)
            text_tokens = (input_ids!=self.model.config.image_token_id).cpu().detach().numpy().reshape(-1)

            if mean_dim!='both':
                # All tokens
                all_layer_embeddings = hidden_states.mean(axis=mean_dim) 
                # Image tokens
                image_embeddings = hidden_states[:,image_tokens,:].mean(axis=mean_dim)
                # Text tokens
                text_embeddings = hidden_states[:,text_tokens,:].mean(axis=mean_dim)
                return all_layer_embeddings, image_embeddings, text_embeddings
            
            else:
                # All tokens
                ax1_all_layer_embeddings = hidden_states.mean(axis=1) 
                ax2_all_layer_embeddings = hidden_states.mean(axis=2) 
                # Image tokens
                ax1_image_embeddings = hidden_states[:,image_tokens,:].mean(axis=1)
                ax2_image_embeddings = hidden_states[:,image_tokens,:].mean(axis=2)
                # Text tokens
                ax1_text_embeddings = hidden_states[:,text_tokens,:].mean(axis=1)
                ax2_text_embeddings = hidden_states[:,text_tokens,:].mean(axis=2)
                
                return ax1_all_layer_embeddings, ax2_all_layer_embeddings, ax1_image_embeddings, ax2_image_embeddings, ax1_text_embeddings, ax2_text_embeddings


    def probing_experiment(self, layer_embeddings, gold_reference_binary, layers='all'):
        """
        Perform probing experiment on the embeddings.

        arguments:
        layer_embeddings: numpy array of shape (n_samples, n_layers, embedding_size)
        gold_reference_binary: binary labels for the probing task
        layers: list of layers to probe, or 'all' for all layers

        returns:
        metrics_dict: dictionary containing the probing metrics for each layer
        """
        print("Probing experiment started...")
        print(f"Shape of layer_embeddings: {layer_embeddings.shape}")
        print(f"Number of samples: {layer_embeddings.shape[0]}, Number of layers: {layer_embeddings.shape[1]}, Embedding size: {layer_embeddings.shape[2]}")


        if layers=='all':
            layers = list(range(layer_embeddings.shape[1]))
        else:
            layers = [layers] if isinstance(layers, int) else layers


        metrics_dict = {'accuracy': [None] * len(layers),
                        'auroc': [None] * len(layers), 
                        'auprc': [None] * len(layers),
                        'f1': [None] * len(layers),
                        'precision': [None] * len(layers),
                        'recall': [None] * len(layers)}

        for idx, layer in tqdm(enumerate(layers), desc="Probing layer"):
            # Stratified split
            X_train, X_test, y_train, y_test = train_test_split(layer_embeddings[:,layer,:], 
                                                                gold_reference_binary, 
                                                                test_size=0.2, 
                                                                random_state=42, 
                                                                stratify=gold_reference_binary)

            # Train logistic regression
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)

            # Predict probabilities and labels
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            auroc = roc_auc_score(y_test, y_prob)
            auprc = average_precision_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred)
            precsion = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            metrics_dict['accuracy'][idx] = acc
            metrics_dict['auroc'][idx] = auroc
            metrics_dict['auprc'][idx] = auprc
            metrics_dict['f1'][idx] = f1
            metrics_dict['precision'][idx] = precsion
            metrics_dict['recall'][idx] = recall

            
        return metrics_dict