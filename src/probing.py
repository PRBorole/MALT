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
import torch.nn as nn
import torch.optim as optim

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score


# Define single-layer MLP
class ProbeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim==None:
            hidden_dim=input_dim

        self.linear1 = nn.Linear(input_dim, 1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 1)          # Output layer
    def forward(self, x):
        x = self.relu(self.linear1(x))
        return torch.sigmoid(self.linear2(x))
                
class LinearProbe():
    """
    Class to handle probing experiments.
    """
    def __init__(self, model, nobjects=None, task=None, mode=None, target=None, prompt_type=None, layers='all'):
        self.nobjects = nobjects
        self.task = task
        self.target = target
        self.model = model
        self.mode = mode
        self.prompt_type = prompt_type
        self.patience = 10
        self.epochs = 100
        self.lr = 1e-3

    def get_prompt(self, ds):
        """        
        Get the prompt for the probing experiment.

        arguments:
        ds: dataset entry containing the scene information

        returns:
        prompt: string containing the prompt for the probing task
        """
        
        
        if self.mode == 'image':
            objects = [{key: data[key] for key in data.keys() if key not in self.task} for data in ds['scene']['objects']]
            condition = ' '.join(
                            [f"{t} {ta} and" 
                            if idx!=len(self.task)-1 else f"{t} {ta}" 
                            for idx,(t,ta) in enumerate(zip(self.task,self.target))]
                        )
            
        elif self.mode=='image_and_text' or self.mode=='count' or self.mode=='text':
            objects = [ds['scene']['objects']]
            condition = ' '.join(
                            [f"{t} {ta} and" 
                            if idx!=len(self.task)-1 else f"{t} {ta}" 
                            for idx,(t,ta) in enumerate(zip(self.task,self.target))]
                        )
            
        if 'image' in self.mode or 'text' in self.mode:
            if self.prompt_type=='complex':
                prompt = "USER: \nThis image shows a minimalist arrangement of 3D geometric shapes made of different materials and colors." +\
                        "The JSON provided contains information about all objects in the scene (material: metal = shinny, rubber = matte))."+\
                        "\n\nUnderstanding the Coordinate System X, Y, Z:"+\
                        "\n• X (Depth): Represents the depth relative to the camera. Smaller values indicate objects that are farther away."+\
                        "\n• Y (Horizontal Position): Represents the left-to-right position. A value of zero means the object is centered in the scene, negative values place the object to the left, and positive values to the right."+\
                        "\n• Z (Vertical Position): Represents the height of the object\'s center point. Larger values correspond to higher vertical positions."+\
                        "\n\nHere is the JSON containing details about all objects in the scene in the image:"+\
                        f"\n\n {({'camera_location': ds['scene']['camera_location'], 'objects': objects})}"+\
                        f"\n Always answer in just one word answer YES or NO to the question. Do not add any other information, only answer YES or NO, is an object with property {condition} present in the image? Only answer yes if a single object contains all queried properties, "+\
                        f"else if no such object with all the properties exist, reply NO ASSISTANT: "
                # Now only answer YES or NO, is an object of color gray and shape cylinder present in the image? ASSISTANT:
            elif self.prompt_type=='simple':
                prompt = f"USER: \n Always answer in just one word answer YES or NO to the question. Do not add any other information, only answer YES or NO, is an object of {condition} present in the image? Only answer yes if a single object contains all queried properties, "+\
                         f"else if no such object with all the properties exist, reply NO ASSISTANT: " 
        
        elif 'count' in self.mode:
            if self.prompt_type=='complex':
                prompt = "USER: \nThis image shows a minimalist arrangement of 3D geometric shapes made of different materials and colors." +\
                        "The JSON provided contains information about all objects in the scene (material: metal = shinny, rubber = matte))."+\
                        "\n\nUnderstanding the Coordinate System X, Y, Z:"+\
                        "\n• X (Depth): Represents the depth relative to the camera. Smaller values indicate objects that are farther away."+\
                        "\n• Y (Horizontal Position): Represents the left-to-right position. A value of zero means the object is centered in the scene, negative values place the object to the left, and positive values to the right."+\
                        "\n• Z (Vertical Position): Represents the height of the object\'s center point. Larger values correspond to higher vertical positions."+\
                        "\n\nHere is the JSON containing details about all objects in the scene in the image:"+\
                        f"\n\n {({'camera_location': ds['scene']['camera_location'], 'objects': objects})}"+\
                        f"\n Always answer in just one word to the question. Do not add any other information, only return in number, how many objects are present in the image? ASSISTANT: "
            elif self.prompt_type=='simple':
                prompt = f"USER: \n Always answer in just one word to the question. Do not add any other information, only return in number, how many objects are present in the image? ASSISTANT: "

        return prompt

    def get_layer_llava_embeddings(self, pixel_values, input_ids, attention_mask, output_attentions=False, mean_dim=1, special_token_embeddings=True):
        """
        Get language model per layer embeddings from the llava model.

        arguments:
        pixel_values: tensor of shape (batch_size, channels, height, width) containing the pixel values of the images
        input_ids: tensor of shape (batch_size, sequence_length)
        attention_mask: tensor of shape (batch_size, sequence_length) indicating which tokens are padded
        output_attentions: bool, indicating if attention should be returned
        mean_dim: dimension along which to take the mean of the embeddings, default is 1 for sententce-level embeddings
        special_token_embeddings: bool, indicating embeddings for special tokens pad, bos should be returned or not

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
            image_features = self.model.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy
                )
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
        with torch.no_grad():
            outputs = self.model.language_model(attention_mask=attention_mask,
                                                inputs_embeds=inputs_embeds,
                                                output_hidden_states=True,
                                                output_attentions=output_attentions,
                                                padding=True,
                                                return_dict=True)

            # Extract hidden states from all layers
            hidden_states = torch.vstack(outputs.hidden_states).cpu().detach().numpy()
            image_tokens = (input_ids==self.model.config.image_token_id).cpu().detach().numpy().reshape(-1)
            text_tokens = ((input_ids!=self.model.config.image_token_id)&
                           (input_ids!=self.model.config.pad_token_id)&
                           (input_ids!=1)).cpu().detach().numpy().reshape(-1)
            special_tokens = ((input_ids==self.model.config.pad_token_id)|(input_ids==1)).cpu().detach().numpy().reshape(-1)

            return_dict = {}
            if mean_dim=='None':
                
                # All tokens
                return_dict['all_layer_embeddings'] = hidden_states

                # Image tokens
                return_dict['img_layer_embeddings'] = hidden_states[:,image_tokens,:]

                # Text tokens
                return_dict['text_layer_embeddings'] = hidden_states[:,text_tokens,:]

                if special_token_embeddings:
                    return_dict['special_layer_embeddings'] = hidden_states[:,special_tokens,:].mean(axis=mean_dim)

            elif mean_dim!='both':
                
                # All tokens
                return_dict['all_layer_embeddings'] = hidden_states.mean(axis=mean_dim) 
                # Image tokens
                return_dict['img_layer_embeddings'] = hidden_states[:,image_tokens,:].mean(axis=mean_dim)
                # Text tokens
                return_dict['text_layer_embeddings'] = hidden_states[:,text_tokens,:].mean(axis=mean_dim)

                if special_token_embeddings:
                    return_dict['ax_special_layer_embeddings'] = hidden_states[:,special_tokens,:].mean(axis=mean_dim)
            
            else:
                # All tokens
                return_dict['ax1_all_layer_embeddings'] = hidden_states.mean(axis=1) 
                return_dict['ax2_all_layer_embeddings'] = hidden_states.mean(axis=2) 
                # Image tokens
                return_dict['ax1_img_layer_embeddings'] = hidden_states[:,image_tokens,:].mean(axis=1)
                return_dict['ax2_img_layer_embeddings'] = hidden_states[:,image_tokens,:].mean(axis=2)
                # Text tokens
                return_dict['ax1_text_layer_embeddings'] = hidden_states[:,text_tokens,:].mean(axis=1)
                return_dict['ax2_text_layer_embeddings'] = hidden_states[:,text_tokens,:].mean(axis=2)
                
                if special_token_embeddings:
                    return_dict['ax1_special_layer_embeddings'] = hidden_states[:,special_tokens,:].mean(axis=1)
                    return_dict['ax2_special_layer_embeddings'] = hidden_states[:,special_tokens,:].mean(axis=2)

            if output_attentions:
                return_dict['attentions'] = torch.vstack(outputs.attentions).cpu().detach().numpy()
        return return_dict
    
    def get_layer_saprot_embeddings(self, tokenizer, inputs, mean_dim=1):
        """
        Get hidden representations of the model.

        Argument:
            inputs:  A dictionary of inputs. It should contain keys ["input_ids", "attention_mask", "token_type_ids"].
            reduction: Whether to reduce the hidden states. If None, the hidden states are not reduced. If "mean",
                        the hidden states are averaged over the sequence length.

        Returns:
            hidden_states: A list of tensors. Each tensor is of shape [L, D], where L is the sequence length and D is
                            the hidden dimension.

        For this code refer to https://github.com/westlake-repl/SaProt/blob/main/model/saprot/base.py#L151
        """
        inputs["output_hidden_states"] = True
        with torch.no_grad():
            outputs = self.model.esm(**inputs)

        # Get the index of the first <eos> token
        input_ids = inputs["input_ids"]
        eos_id = tokenizer.eos_token_id
        ends = (input_ids == eos_id).int()
        indices = ends.argmax(dim=-1)

        hidden_states = outputs["hidden_states"]
        hidden_states = np.array([i[0].cpu().detach().numpy() for i in hidden_states])
        
        # for i, idx in enumerate(indices):
        if mean_dim != "both":
            embeddings = hidden_states[:,1:indices,:].mean(mean_dim)
            return embeddings
        else:
            # Need to implement with padding for token
            ax1_embeddings = hidden_states[:,1:indices,:].mean(1)
            ax2_embeddings =  hidden_states[:,1:indices,:].mean(2)
            return ax1_embeddings, ax2_embeddings


    def probing_experiment(self, layer_embeddings, gold_reference, layers='all'):
        """
        Perform probing experiment on the embeddings.

        arguments:
        layer_embeddings: numpy array of shape (n_samples, n_layers, embedding_size)
        gold_reference: labels for the probing task
        layers: list of layers to probe, or 'all' for all layers

        returns:
        metrics_dict: dictionary containing the probing metrics for each layer
        """

        print("Probing experiment started...")
        if type(layer_embeddings)!=dict and layers=='all':
            layers = list(range(layer_embeddings.shape[1]))
            print(f"Number of samples: {layer_embeddings.shape[0]}, Number of layers: {layer_embeddings.shape[1]}, Embedding size: {layer_embeddings.shape[2]}")
        elif type(layer_embeddings)==dict and layers=='all':
            layers = list(range(layer_embeddings['X_train'].shape[1]))
            print(f"Number of samples: {layer_embeddings['X_train'].shape[0]}, Number of layers: {layer_embeddings['X_train'].shape[1]}, Embedding size: {layer_embeddings['X_train'].shape[2]}")
        else:
            layers = [layers] if isinstance(layers, int) else layers
            print(f"Number of samples: {layer_embeddings.shape[0]}, Number of layers: {layer_embeddings.shape[1]}, Embedding size: {layer_embeddings.shape[2]}")

        metrics_dict = {'accuracy': [None] * len(layers),
                        'auroc': [None] * len(layers), 
                        'auprc': [None] * len(layers),
                        'f1': [None] * len(layers),
                        'precision': [None] * len(layers),
                        'recall': [None] * len(layers)}
        
        for idx, layer in tqdm(enumerate(layers), desc="Probing layer"):

            # Stratified split
            if type(layer_embeddings)!=dict:
                X_train, X_test, y_train, y_test = train_test_split(
                    layer_embeddings[:,layer,:], 
                    gold_reference, 
                    test_size=0.2, 
                    random_state=42, 
                    stratify=gold_reference
                )
                classes = set(gold_reference)
            else:
                X_train, X_test, y_train, y_test = layer_embeddings['X_train'], layer_embeddings['X_test'], gold_reference['y_train'], gold_reference['y_test']
                X_train, X_test = X_train[:,layer,:], X_test[:,layer,:]

                classes = set(y_train)

            # Train logistic regression
            if len(classes)>2:
                clf = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
            else:
                clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)

            # Predict probabilities and labels
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

            # # Convert data to torch tensors
            # X_train = torch.tensor(X_train, dtype=torch.float32)
            # y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            # X_test = torch.tensor(X_test, dtype=torch.float32)
            # y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


            # mlp = ProbeMLP(X_train.shape[1])
            # criterion = nn.BCELoss()
            # optimizer = optim.Adam(mlp.parameters(), lr=self.lr)

            # # Train
            # mlp.train()
            # best_loss = float('inf')
            # patience = self.patience
            # counter = 0

            # for epoch in range(self.epochs): 
            #     optimizer.zero_grad()
            #     outputs = mlp(X_train)
            #     loss = criterion(outputs, y_train)
            #     loss.backward()
            #     optimizer.step()

            #      # Early stopping check
            #     if loss.item() < best_loss - 1e-6:
            #         best_loss = loss.item()
            #         counter = 0
            #         best_state = mlp.state_dict()
            #     else:
            #         counter += 1
            #         if counter >= patience:
            #             break
            # # Predict
            # mlp.eval()
            # with torch.no_grad():
            #     y_prob = mlp(X_test).squeeze().numpy()
            #     y_pred = (y_prob > 0.5).astype(int)

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
    


    def probing_count_experiment(self, layer_embeddings, gold_reference, layers='all'):
        """
        Perform probing experiment on the embeddings.

        arguments:
        layer_embeddings: numpy array of shape (n_samples, n_layers, embedding_size)
        gold_reference: labels for the probing task
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
                        'auroc_macro': [None] * len(layers), 
                        'auprc_macro': [None] * len(layers),
                        'f1_macro': [None] * len(layers),
                        'precision_macro': [None] * len(layers),
                        'recall_macro': [None] * len(layers),
                        'auroc_micro': [None] * len(layers), 
                        'auprc_micro': [None] * len(layers),
                        'f1_micro': [None] * len(layers),
                        'precision_micro': [None] * len(layers),
                        'recall_micro': [None] * len(layers)}

        for idx, layer in tqdm(enumerate(layers), desc="Probing layer"):
            # Stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                layer_embeddings[:,layer,:], 
                gold_reference, 
                test_size=0.2, 
                random_state=42, 
                stratify=gold_reference
            )

            # Train logistic regression
            clf = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)

            clf.fit(X_train, y_train)

            # Predict probabilities and labels
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            auroc_macro = roc_auc_score(y_test, y_prob, average='macro', multi_class='ovr')
            auprc_macro = average_precision_score(y_test, y_prob, average='macro')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            precsion_macro = precision_score(y_test, y_pred, average='macro')
            recall_macro = recall_score(y_test, y_pred, average='macro')

            auroc_micro = roc_auc_score(y_test, y_prob, average='micro', multi_class='ovr')
            auprc_micro = average_precision_score(y_test, y_prob, average='micro')
            f1_micro = f1_score(y_test, y_pred, average='micro')
            precsion_micro = precision_score(y_test, y_pred, average='micro')
            recall_micro = recall_score(y_test, y_pred, average='micro')

            metrics_dict['accuracy'][idx] = acc
            metrics_dict['auroc_macro'][idx] = auroc_macro
            metrics_dict['auprc_macro'][idx] = auprc_macro
            metrics_dict['f1_macro'][idx] = f1_macro
            metrics_dict['precision_macro'][idx] = precsion_macro
            metrics_dict['recall_macro'][idx] = recall_macro
            metrics_dict['auroc_micro'][idx] = auroc_micro
            metrics_dict['auprc_micro'][idx] = auprc_micro
            metrics_dict['f1_micro'][idx] = f1_micro
            metrics_dict['precision_micro'][idx] = precsion_micro
            metrics_dict['recall_micro'][idx] = recall_micro

            
        return metrics_dict