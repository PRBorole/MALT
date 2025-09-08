from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from src.utils import *
import pandas as pd
import numpy as np
from torchinfo import summary
from src.probing import *
import time 
import random
import re
import ast
import json
from src.prosst.structure.get_sst_seq import SSTPredictor
from Bio import PDB
from Bio.SeqUtils import seq1

random.seed(42)

task = 'seq' # seq, struc, combined
model_version = 2048 # AF2, PDB
chain = 'VH' # VH, VL VH-VL

if model_version==2048:
    config_path = f"./../.cache/huggingface/hub/models--AI4Protein--ProSST-2048/snapshots/a44140009dcee7de9a2d40098fbcce0eea7146b7/" 
elif model_version==4096:
    config_path = f"./../.cache/huggingface/hub/models--AI4Protein--ProSST-4096/snapshots/8ceb0f1095fbb9b70a908e8f858fe6432ffee2bf/"

results_path = f'./results/ProSST/{task}/linear_probe_result_{model_version}_{chain}.csv'

tokenizer = AutoTokenizer.from_pretrained(config_path, use_safetensors=True, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(config_path, use_safetensors=True, trust_remote_code=True)

device = "cuda"
model.to(device)

# Set probe
probe = LinearProbe(model=model,
                    task=task)

# Read data
processed_df = pd.read_csv('./../datasets/csi_bli/csi_bli/df_csi_bli_processed.csv')
train_test_df = pd.read_csv('./../datasets/csi_bli/csi_bli/csi_bli_train_test_split.csv')

#######################
quantization_predictor = SSTPredictor(structure_vocab_size=model_version)
pdb_path = './../datasets/csi_bli/csi_bli/structures/pdb_symlinks/' + train_test_df['Name'].iloc[0] + '.pdb'
with torch.no_grad():
    tokens_pdb = quantization_predictor.predict_from_pdb(pdb_path)

pdb_parser = PDB.PDBParser(QUIET=True)
structure = pdb_parser.get_structure("protein", pdb_path)
pdb_model = structure[0]

# extract amino acid sequence
seq = []
# extract amino acid coordinates
aa_coords = {"N": [], "CA": [], "C": [], "O": []}
chain = ['A']


for pdb_model in structure:
    for chain in [c for c in pdb_model if c.get_id() in chain_ls]:
        for residue in chain:
            if residue.get_id()[0] == " ":
                seq.append(residue.get_resname())
                for atom_name in aa_coords.keys():
                    atom = residue[atom_name]
                    aa_coords[atom_name].append(atom.get_coord().tolist())
aa_seq = "".join([seq1(aa) for aa in seq])

#######################

# Read encoding for heavy chain
with open('./../datasets/csi_bli/csi_bli/struc_seqs/struc_seq_heavy_train.json', 'r') as f:
    struc_seq_vh_train = json.load(f)
    struc_seq_vh_train_df = pd.DataFrame(
        {'VH': [''.join([aa for aa in seq if aa.isupper()]) for seq in struc_seq_vh_train],
        'VH_encoding': struc_seq_vh_train}
    )

with open('./../datasets/csi_bli/csi_bli/struc_seqs/struc_seq_heavy_test.json', 'r') as f:
    struc_seq_vh_test = json.load(f)
    struc_seq_vh_test_df = pd.DataFrame(
        {'VH': [''.join([aa for aa in seq if aa.isupper()]) for seq in struc_seq_vh_test],
        'VH_encoding': struc_seq_vh_test}
    )

struc_seq_vh_df = pd.concat([struc_seq_vh_train_df, struc_seq_vh_test_df])

# Read encoding for light chain
with open('./../datasets/csi_bli/csi_bli/struc_seqs/struc_seq_light_train.json', 'r') as f:
    struc_seq_vl_train = json.load(f)
    struc_seq_vl_train_df = pd.DataFrame(
        {'VL': [''.join([aa for aa in seq if aa.isupper()]) for seq in struc_seq_vl_train],
        'VL_encoding': struc_seq_vl_train}
    )

with open('./../datasets/csi_bli/csi_bli/struc_seqs/struc_seq_light_test.json', 'r') as f:
    struc_seq_vl_test = json.load(f)
    struc_seq_vl_test_df = pd.DataFrame(
        {'VL': [''.join([aa for aa in seq if aa.isupper()]) for seq in struc_seq_vl_test],
        'VL_encoding': struc_seq_vl_test}
    )

struc_seq_vl_df = pd.concat([struc_seq_vl_train_df, struc_seq_vl_test_df])

# Combined VH-VL
struc_seq_vh_vl_train_df = pd.DataFrame(
    {'VH-VL': [vh+'-'+vl for vh,vl in zip(struc_seq_vh_train_df['VH'].to_list(), struc_seq_vl_train_df['VL'].to_list())],
    'VH-VL_encoding': [vh+'##'+vl for vh,vl in zip(struc_seq_vh_train_df['VH_encoding'].to_list(), struc_seq_vl_train_df['VL_encoding'].to_list())]}
)

struc_seq_vh_vl_test_df = pd.DataFrame(
    {'VH-VL': [vh+'-'+vl for vh,vl in zip(struc_seq_vh_test_df['VH'].to_list(), struc_seq_vl_test_df['VL'].to_list())],
    'VH-VL_encoding': [vh+'##'+vl for vh,vl in zip(struc_seq_vh_test_df['VH_encoding'].to_list(), struc_seq_vl_test_df['VL_encoding'].to_list())]}
)

struc_seq_vh_vl_df = pd.concat([struc_seq_vh_vl_train_df, struc_seq_vh_vl_test_df])

chain_dict = {'VH': struc_seq_vh_df,
            'VL': struc_seq_vl_df,
            'VH-VL': struc_seq_vh_vl_df}


###### chain results
chain_df = train_test_df.merge(chain_dict[chain], on=chain)[['Train', 
                                                        'Test', 
                                                        f'{chain}_encoding',
                                                        'Signal Normalised to -ve Control_binarized']].drop_duplicates(f'{chain}_encoding')

X_train = chain_df[chain_df['Train']==1][f'{chain}_encoding'].to_list()
X_test = chain_df[chain_df['Test']==1][f'{chain}_encoding'].to_list()
y_train = chain_df[chain_df['Train']==1]['Signal Normalised to -ve Control_binarized'].to_list()
y_test = chain_df[chain_df['Test']==1]['Signal Normalised to -ve Control_binarized'].to_list()

if task=='seq':
    X_train = [''.join([aa if aa.isupper() else '#' for aa in seq]) for seq in X_train]
    X_test = [''.join([aa if aa.isupper() else '#' for aa in seq]) for seq in X_test] 
elif task=='struc':
    X_train = [''.join(['#' if aa.isupper() else aa for aa in seq]) for seq in X_train]
    X_test = [''.join(['#' if aa.isupper() else aa for aa in seq]) for seq in X_test]

s_train_embeddings = [None]*len(X_train)
# t_train_embeddings = [None]*len(X_train)

for idx,seq in tqdm(enumerate(X_train)):
    tokens = tokenizer.tokenize(seq)
    inputs = tokenizer(seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    embeddings = probe.get_layer_saprot_embeddings(
        inputs=inputs, 
        tokenizer=tokenizer,
        mean_dim=1
    )
    s_train_embeddings[idx] = embeddings
    # t_train_embeddings[idx] = embeddings[1]

s_train_embeddings = np.array(s_train_embeddings)
# t_train_embeddings = np.array(t_train_embeddings)

s_test_embeddings = [None]*len(X_test)
# t_test_embeddings = [None]*len(X_test)

for idx,seq in tqdm(enumerate(X_test)):
    tokens = tokenizer.tokenize(seq)
    inputs = tokenizer(seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    embeddings = probe.get_layer_saprot_embeddings(
        inputs=inputs, 
        tokenizer=tokenizer,
        mean_dim=1
    )
    s_test_embeddings[idx] = embeddings
    # t_test_embeddings[idx] = embeddings[1]

s_test_embeddings = np.array(s_test_embeddings)
# t_test_embeddings = np.array(t_test_embeddings)

layer_embeddings = {
    'X_train':s_train_embeddings,
    'X_test':s_test_embeddings,
    }

gold_reference = {
    'y_train':y_train,
    'y_test':y_test
    }

# Probing experiment token text 
s_metrics = probe.probing_experiment(layer_embeddings=layer_embeddings,
                                    gold_reference=gold_reference)
s_metrics= pd.DataFrame(s_metrics)
s_metrics['embedding_level'] = ['sentence_text'] * len(s_metrics)
s_metrics['layer'] = list(range(s_metrics.shape[0]))
s_metrics['chain'] = [chain]*len(s_metrics)
s_metrics.to_csv(results_path)

