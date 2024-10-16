import warnings
warnings.filterwarnings('ignore')
import csv
import json
import argparse
import sys
import torch
import pandas as pd
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from trl import setup_chat_format
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

#============================================
#                ARGPARSE
#============================================
parser = argparse.ArgumentParser(description='Configuration for model training.')
parser.add_argument('--lang', type=str, choices=['ITA', 'ENG'], required=True, help='Language of the prompt: "ITA" or "ENG".')
parser.add_argument('--prompt_number', type=int, choices=[1, 2], required=True, help='Number of the prompt to use: 1 or 2.')
parser.add_argument('--adapter', type=str, help='Adapter to use.')
parser.add_argument('--base_model', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='Base model to use. Default: "meta-llama/Meta-Llama-3-8B-Instruct".')
parser.add_argument('--adddoc', type=int, default=1, help='if Use document name. 1:Use 0:no ')
parser.add_argument('--splitevidence', type=int, default=0, help='if split evidence. 1:yes 0:no ')
# Parse command line arguments
args = parser.parse_args()

# Assign arguments to variables
language = args.lang
prompt_number = args.prompt_number
adapter = args.adapter
adddoc=args.adddoc
splitevidence=args.splitevidence
BASE_MODEL = args.base_model

# Check if the arguments are valid
if (prompt_number not in [1, 2]) or (language not in ["ITA", "ENG"]) or not adapter:
    print("Error: The program must be launched with the following parameters:")
    print("--lang with values 'ITA' or 'ENG'")
    print("--prompt_number with values 1 or 2")
    print("--adapter with the adapter name. Example: --adapter adapter")
    print("--base_model with the model name. Example: --base_model 'meta-llama/Meta-Llama-3-8B-Instruct")
    print("prompt 1: Defines the task")
    print("prompt 2: Defines the task and provides examples")
    print("Example: python test_baseline.py --lang ENG --prompt_number 2 --adapter 'NO' --base_model 'meta-llama/Meta-Llama-3-8B-Instruct'")
    sys.exit(1)

#============================================
#       PROMPT FUNCTIONS AND STRINGS
#============================================

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

def format_chat_template_inf(claim, evidence):
    user_prompt = generate_prompt_str(claim, evidence,adddoc)

    messages = [{"role": "system", "content": selected_prompt},
                {"role": "user", "content": user_prompt}]
    messages = tokenizer.apply_chat_template(messages, tokenize=False)
    return messages

def read_json_test(file_path,splitevidence):
    inputs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        if splitevidence:
            for line in file:
                record = json.loads(line)
                claim = record.get("claim", "")
                label = record.get("label", "")
                evidence = record.get("evidence", "")

                if isinstance(evidence, list) and len(evidence) > 1:
                    for ev in evidence:
                        inputs.append([claim, label, [ev]])
                else:
                    inputs.append([claim, label, evidence])
        else:

            for line in file:
                record = json.loads(line)
                claim = record.get("claim", "")
                label = record.get("label", "")
                evidence = record.get("evidence", "")

                inputs.append([claim, label, evidence])
    return inputs

#==========================================================================
#       ADD ADAPTER TO BASE MODEL AND INFERENCE FUNCTIONS
#==========================================================================
def merge_model(base_model, adapter_path):
  model = AutoModelForCausalLM.from_pretrained(
    base_model,
    # return_dict=True,
    # low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
    # trust_remote_code=True,
  )

  # Load tokenizer
  tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL,  padding_side="left")
  tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
  model, tokenizer = setup_chat_format(model, tokenizer)

  # Peft model with LoRA
  '''
  model = PeftModel.from_pretrained(
    model,
    adapter_path,
    torch_dtype=torch.float16,
    device_map="auto",
  )
  '''
  return model,tokenizer

def inference(prompt_with_claim,tokenizer,model):
  inputs = tokenizer(prompt_with_claim, return_tensors='pt', padding=True, truncation=True).to("cuda")
  generation_config = model.generation_config
  generation_config.pad_token_id = tokenizer.eos_token_id
  generation_config.do_sample=False
  shape = inputs["input_ids"].shape

  num_values = shape[0] * shape[1]

  
  outputs = model.generate(**inputs, max_length=num_values+7, temperature=0.1, generation_config=generation_config)
  text = tokenizer.decode(outputs[0], skip_special_tokens=True)

  
  if language == "ENG":
        return text.split("### Answer:")[-1].replace("\n", "")
  elif language == "ITA":
        return text.split("### Risposta:")[-1].replace("\n", "") #
    


#============================================
#              INITIALIZE  
#============================================ 

bits = "full"
data_path = "."
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
test_file = f"{data_path}/test_{language}_GOLD.jsonl"

CUT_INPUT_CHAR_LENGTH = 1200					# Maximum character length for input data

inputs = read_json_test(test_file,splitevidence)
model,tokenizer = merge_model(BASE_MODEL,f"{adapter}")
tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
import time
#============================================
#              EXEC TEST  
#============================================ 
true_labels = []
predicted_labels = []
claims = []
evidences = []
total_instances = len(inputs)  
counter = 0
print("input example:",str(total_instances))
from datetime import datetime
start_time = datetime.now()
print("Start time:", start_time)
for input in inputs:
  claim = input[0]
  claims.append(claim)
  expected_output = input[1]
  true_labels.append(expected_output)
  evidence = input[2]
  evidences.append(evidence)

  txt = format_chat_template_inf(claim, evidence)
  output = inference(txt,tokenizer,model)
  
  print("\n\n\n")
  print(output)
  print("***************************************")

  if len(output) > 40:
    #print(f"Error in generation. Output: {output}")
    zz=1
    #output = "NOT ENOUGH INFO"
  if "SUPPORTS" in output:
      output = "SUPPORTS"
  elif "REFUTES" in output:
      output = "REFUTES"
  elif "NOT ENOUGH INFO" in output:
      output = "NOT ENOUGH INFO"
  else:
      output = "NOT ENOUGH INFO"    
      print("*****************INCORRECT**********************")
  predicted_labels.append(output)
  
  counter += 1
  if counter % 50 == 0:
      print(f"Processed {counter} instances")
#============================================
#          SAVE PREDICTIONS IN FILE  
#============================================		
with open("GOLD_BEST_BASELINE_"+language+"_pr_"+str(prompt_number)+"_ADD_"+str(adddoc)+"_SPLIT_"+str(splitevidence)+'output_pred_llm.jsonl', 'w') as output_pred_llm:
    for i in range(total_instances):
        json.dump({'claim': claims[i], 'evidence': evidences[i], 'pred_label': predicted_labels[i], 'true_label': true_labels[i]}, output_pred_llm)
        output_pred_llm.write('\n')

#============================================
#              EVALUATE TEST  
#============================================
accuracy = accuracy_score(true_labels, predicted_labels)

precision_supp = precision_score(true_labels, predicted_labels, labels=["SUPPORTS"], average='macro')
recall_supp = recall_score(true_labels, predicted_labels, labels=["SUPPORTS"], average='macro')
f1_supp = 2 * (precision_supp * recall_supp) / (precision_supp + recall_supp)

precision_ref = precision_score(true_labels, predicted_labels, labels=["REFUTES"], average='macro')
recall_ref = recall_score(true_labels, predicted_labels, labels=["REFUTES"], average='macro')
f1_ref = 2 * (precision_ref * recall_ref) / (precision_ref + recall_ref)

precision_nei = precision_score(true_labels, predicted_labels, labels=["NOT ENOUGH INFO"], average='macro')
recall_nei = recall_score(true_labels, predicted_labels, labels=["NOT ENOUGH INFO"], average='macro')
f1_nei = 2 * (precision_nei * recall_nei) / (precision_nei + recall_nei)

weighted_f1 = (f1_supp + f1_ref + f1_nei) / 3

conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])

end_time = datetime.now()
print("End time:", end_time)
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time)

print("GOLD_BEST_BASELINE_"+language+"_pr_"+str(prompt_number)+"_ADD_"+str(adddoc)+"_SPLIT_"+str(splitevidence))
print(f"Number of instances: {total_instances}")

print("\nAccuracy:", round(accuracy,4))

print("\nPrecision e recall SUPPORTS:")
print("Recall:", round(recall_supp,4))
print("Precision:", round(precision_supp,4))

print("\nPrecision e recall REFUTES:")
print("Recall:", round(recall_ref,4))
print("Precision:", round(precision_ref,4))

print("\nPrecision e recall NOT ENOUGH INFO:")
print("Recall:", round(recall_nei,4))
print("Precision:", round(precision_nei,4))

print("\nF1 score SUPPORTS:", round(f1_supp, 4))
print("F1 score REFUTES:", round(f1_ref, 4))
print("F1 score NOT ENOUGH INFO:", round(f1_nei, 4))
print("F1 general score:", round(weighted_f1, 4))

print("\nConfusion Matrix: SUPPORTS, REFUTES, NOT ENOUGH INFO")
print(conf_matrix)
a="GOLD_BEST_BASELINE_"+language+"_pr_"+str(prompt_number)+"_ADD_"+str(adddoc)+"_SPLIT_"+str(splitevidence)+"\t"+str(total_instances)+"\t"+str(round(accuracy,4))+"\t"+str(round(recall_supp,4))+"\t"+str(round(precision_supp,4))+"\t"+str(round(recall_ref,4))+"\t"+str(round(precision_ref,4))+"\t"+str(round(recall_nei,4))+"\t"+str(round(precision_nei,4))+"\t"+str(round(f1_supp, 4))+"\t"+str(round(f1_ref, 4))+"\t"+str(round(f1_nei, 4))+"\t"+str(round(weighted_f1, 4))
print("RESULT\t"+a)
