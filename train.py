# -*- coding: utf-8 -*-

import transformers

import warnings
warnings.filterwarnings('ignore')
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

import csv
import json
import argparse
import sys

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer, setup_chat_format

import torch
from datasets import load_dataset
import pandas as pd
import re

#============================================
#                ARGPARSE
#============================================
parser = argparse.ArgumentParser(description='Configuration for model training.')
parser.add_argument('--lang', type=str, choices=['ITA', 'ENG'], required=True, help='Language of the prompt: "ITA" or "ENG".')
parser.add_argument('--prompt_number', type=int, choices=[1, 2], required=True, help='Number of the prompt to use: 1 or 2.')
parser.add_argument('--learn_rate', type=float, default=0.0001, help='Learning rate for training. Default: 0.0001. Must be a positive number starting from 0.001.')
parser.add_argument('--epochs', type=int, choices=range(1, 11), default=3, help='Number of epochs for training. Default: 3. Must be a positive integer between 1 and 10.')
parser.add_argument('--base_model', type=str, default='meta-llama/Meta-Llama-3-8B', help='Base model to use. Default: "meta-llama/Meta-Llama-3-8B".')
parser.add_argument('--max_seq_length', type=int, default=1024, help='max_seq_length.')
parser.add_argument('--conf', type=str, default="config.json", help='config file.')
parser.add_argument('--adddoc', type=int, default=1, help='if Use document name. 1:Use 0:no ')
parser.add_argument('--splitevidence', type=int, default=0, help='if split evidence. 1:yes 0:no ')

# Parse command line arguments
args = parser.parse_args()

# Assign arguments to variables
language = args.lang
prompt_number = args.prompt_number
BASE_MODEL = args.base_model
LEARNING_RATE = args.learn_rate
EPOCHS = args.epochs
max_seq_length = args.max_seq_length
confpath = args.conf
adddoc=args.adddoc
splitevidence=args.splitevidence
add_eos_token = True

# Check if the arguments are valid
if (prompt_number not in [1, 2]) or (language not in ["ITA", "ENG"]):
    print("Error: The program must be launched with the following parameters:")
    print("--lang with values 'ITA' or 'ENG'")
    print("--prompt_number with values 1 or 2")
    print("--learn_rate with a positive number starting from 0.001. Example: --learn_rate 0.0001")
    print("--epochs with a positive integer between 1 and 10. Example: --epochs 3")
    print("--base_model with the model name. Example: --base_model meta-llama/Meta-Llama-3-8B")
    print("prompt 1: Defines the task")
    print("prompt 2: Defines the task and provides examples")
    print("Example: python train.py --lang ENG --prompt_number 2 --learn_rate 0.0001 --epochs 3 --base_model meta-llama/Meta-Llama-3-8B")
    sys.exit(1)

if not BASE_MODEL:
    print("No base model specified. The default model 'meta-llama/Meta-Llama-3-8B' will be used.")


""" Task definition, explanation of output labels, presentation of three examples, and indication of the title of the document from which the evidence is drawn. """
prompt1_ENG_doc = f"""### Instruction
Evaluate if the claim is supported by the evidence provided. Definitions for key terms used in this task are:
- Claim: A statement or assertion under examination.
- Evidence: Information that either supports or opposes the claim.
- Document: denotes the source document for the evidence.

Answer with one of the following judgments based on the evidence provided:
- SUPPORTS: if the evidence substantiates the claim.
- REFUTES: if the evidence directly contradicts the claim.
- NOT ENOUGH INFO: if there is insufficient evidence to determine the claim's validity
"""

prompt2_ENG_doc  = f"""### Instruction
Evaluate if the claim is supported by the evidence provided. Definitions for key terms used in this task are:
- Claim: A statement or assertion under examination.
- Evidence: Information that either supports or opposes the claim.
- Document: denotes the source document for the evidence.

Answer with one of the following judgments based on the evidence provided:
- SUPPORTS: if the evidence substantiates the claim.
- REFUTES: if the evidence directly contradicts the claim.
- NOT ENOUGH INFO: if there is insufficient evidence to determine the claim's validity

### Examples
These examples demonstrate how to apply the evaluation criteria:
- Claim: The Germanic peoples are also called Gothic.
- Evidence: The Germanic peoples (also referred to as Teutonic, Suebian, or Gothic in older literature) are an Indo-European ethno-linguistic group of Northern European origin.
- Document: Germanic peoples
- Answer: SUPPORTS

- Claim: Tennis is not a sport.
- Evidence: Tennis is played by millions of recreational players and is also a popular worldwide spectator sport.
- Document: Tennis
- Answer: REFUTES

- Claim: Kick-Ass is a horror film.
- Evidence: Kick-Ass is a 2010 British-American film based on the comic book of the same name by Mark Millar and John Romita, Jr.
- Document: Kick-Ass (film)
- Answer: NOT ENOUGH INFO
"""

""" Task definition, explanation of output labels, presentation of three examples, and indication of the title of the document from which the evidence is drawn. """
prompt1_ITA_doc  = f"""### Istruzioni
Valuta se l'affermazione è supportata dalle prove fornite. Le definizioni dei termini chiave utilizzati in questo compito sono:
- Affermazione: Una dichiarazione o asserzione sotto esame.
- Prova: Informazioni che supportano o contraddicono l'affermazione.
- Documento: indica la fonte da cui è stata estratta la prova.

Rispondi con uno dei seguenti giudizi basati sulle prove fornite:
- SUPPORTS: se le prove confermano l'affermazione.
- REFUTES: se le prove contraddicono direttamente l'affermazione.
- NOT ENOUGH INFO: se le prove non sono sufficienti per determinare la validità dell'affermazione.
"""

prompt2_ITA_doc  = f"""### Istruzioni
Valuta se l'affermazione è supportata dalle prove fornite. Le definizioni dei termini chiave utilizzati in questo compito sono:
- Affermazione: Una dichiarazione o asserzione sotto esame.
- Prova: Informazioni che supportano o contraddicono l'affermazione.
- Documento: indica la fonte da cui è stata estratta la prova.

Rispondi con uno dei seguenti giudizi basati sulle prove fornite:
- SUPPORTS: se le prove confermano l'affermazione.
- REFUTES: se le prove contraddicono direttamente l'affermazione.
- NOT ENOUGH INFO: se le prove non sono sufficienti per determinare la validità dell'affermazione.

### Esempi
Questi esempi dimostrano come applicare i criteri di valutazione:
- Affermazione: I popoli germanici sono chiamati anche gotici.
- Prova: I popoli germanici (anche chiamati Teutoni, Suebi o Goti nella letteratura più antica) sono un gruppo etno-linguistico indoeuropeo di origine nord europea.
- Documento: Popoli germanici
- Risposta: SUPPORTS

- Affermazione: Il tennis non è uno sport.
- Prova: Il tennis è praticato da milioni di giocatori amatoriali ed è anche uno sport popolare a livello mondiale.
- Documento: Tennis
- Risposta: REFUTES

- Affermazione: Kick-Ass è un film horror.
- Prova: Kick-Ass è un film britannico-americano del 2010 basato sul fumetto omonimo di Mark Millar e John Romita Jr.
- Documento: Kick-Ass (film)
- Risposta: NOT ENOUGH INFO
"""

prompt1_ENG = f"""### Instruction
Evaluate if the claim is supported by the evidence provided. Definitions for key terms used in this task are:
- Claim: A statement or assertion under examination.
- Evidence: Information that either supports or opposes the claim.

Answer with one of the following judgments based on the evidence provided:
- SUPPORTS: if the evidence substantiates the claim.
- REFUTES: if the evidence directly contradicts the claim.
- NOT ENOUGH INFO: if there is insufficient evidence to determine the claim's validity
"""

prompt2_ENG = f"""### Instruction
Evaluate if the claim is supported by the evidence provided. Definitions for key terms used in this task are:
- Claim: A statement or assertion under examination.
- Evidence: Information that either supports or opposes the claim.

Answer with one of the following judgments based on the evidence provided:
- SUPPORTS: if the evidence substantiates the claim.
- REFUTES: if the evidence directly contradicts the claim.
- NOT ENOUGH INFO: if there is insufficient evidence to determine the claim's validity

### Examples
These examples demonstrate how to apply the evaluation criteria:
- Claim: The Germanic peoples are also called Gothic.
- Evidence: The Germanic peoples (also referred to as Teutonic, Suebian, or Gothic in older literature) are an Indo-European ethno-linguistic group of Northern European origin.
- Answer: SUPPORTS

- Claim: Tennis is not a sport.
- Evidence: Tennis is played by millions of recreational players and is also a popular worldwide spectator sport.
- Answer: REFUTES

- Claim: Kick-Ass is a horror film.
- Evidence: Kick-Ass is a 2010 British-American film based on the comic book of the same name by Mark Millar and John Romita, Jr.
- Answer: NOT ENOUGH INFO
"""

""" Task definition, explanation of output labels, presentation of three examples, and indication of the title of the document from which the evidence is drawn. """
prompt1_ITA = f"""### Istruzioni
Valuta se l'affermazione è supportata dalle prove fornite. Le definizioni dei termini chiave utilizzati in questo compito sono:
- Affermazione: Una dichiarazione o asserzione sotto esame.
- Prova: Informazioni che supportano o contraddicono l'affermazione.

Rispondi con uno dei seguenti giudizi basati sulle prove fornite:
- SUPPORTS: se le prove confermano l'affermazione.
- REFUTES: se le prove contraddicono direttamente l'affermazione.
- NOT ENOUGH INFO: se le prove non sono sufficienti per determinare la validità dell'affermazione.
"""

prompt2_ITA = f"""### Istruzioni
Valuta se l'affermazione è supportata dalle prove fornite. Le definizioni dei termini chiave utilizzati in questo compito sono:
- Affermazione: Una dichiarazione o asserzione sotto esame.
- Prova: Informazioni che supportano o contraddicono l'affermazione.

Rispondi con uno dei seguenti giudizi basati sulle prove fornite:
- SUPPORTS: se le prove confermano l'affermazione.
- REFUTES: se le prove contraddicono direttamente l'affermazione.
- NOT ENOUGH INFO: se le prove non sono sufficienti per determinare la validità dell'affermazione.

### Esempi
Questi esempi dimostrano come applicare i criteri di valutazione:
- Affermazione: I popoli germanici sono chiamati anche gotici.
- Prova: I popoli germanici (anche chiamati Teutoni, Suebi o Goti nella letteratura più antica) sono un gruppo etno-linguistico indoeuropeo di origine nord europea.
- Risposta: SUPPORTS

- Affermazione: Il tennis non è uno sport.
- Prova: Il tennis è praticato da milioni di giocatori amatoriali ed è anche uno sport popolare a livello mondiale.
- Risposta: REFUTES

- Affermazione: Kick-Ass è un film horror.
- Prova: Kick-Ass è un film britannico-americano del 2010 basato sul fumetto omonimo di Mark Millar e John Romita Jr.
- Risposta: NOT ENOUGH INFO
"""

# Function to process a sentence by substituting certain patterns
def process_sent(sentence):
    sentence = re.sub(" LSB.*?RSB", "", sentence)
    sentence = re.sub("-LRB-", "(", sentence)
    sentence = re.sub("LRB", "(", sentence)
    sentence = re.sub("-RRB-", ")", sentence)
    sentence = re.sub("RRB", ")", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)
    sentence = clean_dirty_string(sentence)
    return sentence

# Function to process a Wikipedia title by substituting certain patterns
def process_wiki_title(title):
    title = re.sub("_", " ", title)
    title = re.sub("-LRB-", "(", title)
    title = re.sub("LRB", "(", title)
    title = re.sub("-RRB-", ")", title)
    title = re.sub("RRB", ")", title)
    title = re.sub("COLON", ":", title)
    return title
    
def clean_dirty_string(str):  
    """ Clean dirty content in corpus """
    str = str.replace("-LSB-", "")
    str = str.replace("-RSB", "")
    str = str.replace("  ", " ")
    str = str.replace('( ', '(')
    str = str.replace(' )', ')')
    str = str.replace(' .', '.')
    str = str.replace(' ,', ',')
    str = str.replace(' :', ':')
    str = str.replace(' ;', ';')
    str = str.replace(' -', '')
    str = str.replace(' !', '!')
    str = str.replace(' ?', '?')
    str = str.replace(" n't", "n't")
    str = str.replace(" 's", "'s")
    str = str.replace("\'s", "'s")
    str = str.replace(" ` ", "")
    str = str.replace(" .", ".")
    str = str.replace(" ,", ",")
    str = str.replace("  ", " ")
    return str

# Select the prompt based on the input number
if language == "ENG":
    if prompt_number == 1:
            if adddoc == 1:
                selected_prompt = prompt1_ENG_doc
            else:
                selected_prompt = prompt1_ENG
    elif prompt_number == 2:
            if adddoc == 1:
                selected_prompt = prompt2_ENG_doc
            else:
                selected_prompt = prompt2_ENG

elif language == "ITA":
    if prompt_number == 1:
            if adddoc == 1:
                selected_prompt = prompt1_ITA_doc
            else:
                selected_prompt = prompt1_ITA

    elif prompt_number == 2:
            if adddoc == 1:
                selected_prompt = prompt2_ITA_doc
            else:
                selected_prompt = prompt2_ITA        
        
# Function to merge evidence sentences into a single string
def mergeEvidence(evidence):
    merged = ""
    for ev in evidence:
        merged+=" "
        merged+=process_sent(ev[2])

    return merged

# Function to extract the title of the document from evidence
def get_documTitle(evidence):
  return process_wiki_title(evidence[0])

def prepare(ev):
    procev = process_sent(ev[2])
    return procev.strip()
# Function to generate a prompt for string formatting
def generate_prompt_str(claim, evidence,adddoc): #{selected_prompt}

    prompt = ""
    if adddoc:
        if language == "ITA":
            evidence_str = ""
            for ev in evidence:
                evidence_str += f"- Prova: {prepare(ev)}\n- Documento: {get_documTitle(ev)}\n"
            prompt = f"""### Input
- Affermazione: {claim}
{evidence_str}
### Risposta: """
        else:  # ENG
            evidence_str = ""
            for ev in evidence:
                evidence_str += f"- Evidence: {prepare(ev)}\n- Document: {get_documTitle(ev)}\n"
            prompt = f"""### Input
- Claim: {claim}
{evidence_str}
### Answer: """
    else:
        if language == "ITA":
            evidence_str = ""
            for ev in evidence:
                evidence_str += f"- Prova: {prepare(ev)}\n"
            prompt = f"""### Input
- Affermazione: {claim}
{evidence_str}
### Risposta: """
        else:  # ENG
            evidence_str = ""
            for ev in evidence:
                evidence_str += f"- Evidence: {prepare(ev)}\n"
            prompt = f"""### Input
- Claim: {claim}
{evidence_str}
### Answer: """          

    return prompt

#============================================
#          TRAIN FUNCTIONS
#============================================
# LOAD INPUT JSONL files in the new format
def load(input_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    dataset_df = pd.DataFrame(data)
    return dataset_df

# Function to load and prepare data in the new format
def load_and_prepare_data(input_file_path: str,splitevidence):
    print(input_file_path)
    if splitevidence:
            for row_dict in df.to_dict(orient="records"):
                evidence_list = row_dict["evidence"]        
                # Check if evidence contains more than one element
                if len(evidence_list) > 1:
                    for evidence in evidence_list:
                        dataset_data.append({
                            "claim": row_dict["claim"],  # Assuming the first sentence is the input
                            "evidence": [evidence],  # Use the individual evidence item
                            "output": row_dict["label"]
                        })
                else:
                    dataset_data.append({
                        "claim": row_dict["claim"],  # Assuming the first sentence is the input
                        "evidence": evidence_list,  # Use the single evidence item
                        "output": row_dict["label"]
                    })

            return dataset_data

    else :
            df = load(input_file_path)
            dataset_data = [
                {
                    "claim": row_dict["claim"],  # Assuming the first sentence is the input
                    "evidence": row_dict["evidence"],  # Assuming the first sentence is the input
                    "output": row_dict["label"]
                }
                for row_dict in df.to_dict(orient="records")
            ]
            return dataset_data

def tokenize(prompt, cutoff_len, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def format_chat_template(row):
    user_prompt = generate_prompt_str(row["claim"], row["evidence"],adddoc)
    row_json = [{"role": "system", "content": selected_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": row["output"]}]

    full_prompt = tokenizer.apply_chat_template(row_json, tokenize=False)
    tokenized_full_prompt = tokenize(full_prompt, max_seq_length)

    row_json_input = [{"role": "system", "content": selected_prompt},
                {"role": "user", "content": user_prompt}]
    user_prompt = tokenizer.apply_chat_template(row_json_input, tokenize=False) 
    tokenized_user_prompt = tokenize(
        user_prompt, max_seq_length, add_eos_token=add_eos_token
    )
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    if add_eos_token:
        user_prompt_len -= 1

    tokenized_full_prompt["labels"] = [
        -100
    ] * user_prompt_len + tokenized_full_prompt["labels"][
        user_prompt_len:
    ]  
    return tokenized_full_prompt




# Function to trim long input data
def trim_long_input(json_input, cutoff_len=10000000):
    for json_data in json_input:
        json_data["evidence"] = json_data["evidence"][:cutoff_len]
    return json_input
def trim_long_input(json_input, cutoff_len=10000000):
    cut_count = 0
    for json_data in json_input:
        if len(json_data["evidence"]) > cutoff_len:
            json_data["evidence"] = json_data["evidence"][:cutoff_len]
            cut_count += 1
            
    return json_input, cut_count

#============================================
#               CONFIGURATIONS
#============================================
bits = "4"							                       
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"	

data_path = "."							                                                                        
input_train_path = f"{data_path}/train_{language}.jsonl"		                                                
input_dev_path = f"{data_path}/valid_{language}.jsonl"			                                               
OUTPUT_DIR = f"{data_path}/BEST_adapter_{language}_pr{prompt_number}_add{adddoc}__split{splitevidence}_ep{EPOCHS}_lr{LEARNING_RATE}_{BASE_MODEL}"	    # Output directory

CUT_INPUT_CHAR_LENGTH = 1200					# Maximum character length for input data

with open(confpath, 'r') as config_file:
    config = json.load(config_file)

LORA_R = config["LORA_R"]
LORA_ALPHA = config["LORA_ALPHA"]
LORA_DROPOUT = config["LORA_DROPOUT"]
LORA_TARGET_MODULES = config["LORA_TARGET_MODULES"]
BATCH_SIZE = config["BATCH_SIZE"]
MICRO_BATCH_SIZE = config["MICRO_BATCH_SIZE"]
GRADIENT_ACCUMULATION_STEPS = config["GRADIENT_ACCUMULATION_STEPS"]
WARMUP_RATIO = config["WARMUP_RATIO"]

tmp_train_file_name = "BEST_"+language+"_pr_"+str(prompt_number)+"_ADD_"+str(adddoc)+"_SPLIT_"+str(splitevidence)+"_tmp_train.json"				# Temporary file name for training data
tmp_dev_file_name = "BEST_"+language+"_pr_"+str(prompt_number)+"_ADD_"+str(adddoc)+"_SPLIT_"+str(splitevidence)+"_tmp_dev.json"				    # Temporary file name for validation data

adddoc=args.adddoc
splitevidence=args.splitevidence
#============================================
#               LOAD MODEL
#============================================
torch_dtype = torch.bfloat16

print("LOAD MODEL")
# Load model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, device_map=DEVICE)

print("LOAD TOK")
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
print("SETUP CHAT FORMAT")
model, tokenizer = setup_chat_format(model, tokenizer)

#LORA config
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print("LOAD DATA")

#============================================
#            	LOAD DATA
#============================================
# Load and prepare training and validation data
train_data = load_and_prepare_data(input_train_path,splitevidence)

dev_data = load_and_prepare_data(input_dev_path,splitevidence)

# Write data to temporary files
with open(tmp_train_file_name, "w") as f:
   json.dump(train_data, f)
with open(tmp_dev_file_name, "w") as f:
   json.dump(dev_data, f)

# Load data using datasets library
json_train = load_dataset("json", data_files=tmp_train_file_name)
json_dev = load_dataset("json", data_files=tmp_dev_file_name)
print("SHUFFLE DATA")

json_train = json_train.shuffle(seed=65)
json_dev = json_dev.shuffle(seed=65)

# Trim long input data
json_train["train"],cut_count_t  = trim_long_input(json_train["train"], CUT_INPUT_CHAR_LENGTH)
json_dev["train"],cut_count_v = trim_long_input(json_dev["train"], CUT_INPUT_CHAR_LENGTH)
print(f"Numero di stringhe tagliate train: {cut_count_t}")
print(f"Numero di stringhe tagliate val: {cut_count_v}")

print("MAP DATA")

# Apply the transformation to both training and development sets and store them separately

train_dataset = json_train.map(
    format_chat_template,
    num_proc=1,
)

dev_dataset = json_dev.map(
    format_chat_template,
    num_proc=1,
)
#============================================
#            TRAIN PARAMETERS
#============================================
print("Define training_arguments")
training_arguments = TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=WARMUP_RATIO,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_strategy="steps",
    logging_steps=1,
    optim="adamw_torch",
    evaluation_strategy="epoch",  
    # eval_steps=100,           
    save_strategy="epoch",
    output_dir=OUTPUT_DIR,
    #save_total_limit=1,
    load_best_model_at_end=True,
    label_names=["labels"],
    report_to=[]
)
print("Define trainer")

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

# istantiate a Trainer object using the hyper-params we defined earlier
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset["train"],
    eval_dataset=dev_dataset["train"],
    args=training_arguments,
    data_collator=data_collator
)
model.config.use_cache = False


#============================================
#            TRAIN AND SAVE
#============================================
print(f"Start training. Output: {OUTPUT_DIR}")

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained(OUTPUT_DIR)

print(f"Training completed. Output: {OUTPUT_DIR}")