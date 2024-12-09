# FEVER-it
This repository contains the dataset and code for "Leveraging Large Language Models for Fact Verification in Italian" published at CLiC-it 2024 by Antonio Scaiella, Stefano Costanzo, Elisa Passone, Danilo Croce, and Giorgio Gambosi. The paper is available [HERE](https://clic2024.ilc.cnr.it/wp-content/uploads/2024/12/97_main_long.pdf)

## Dataset
FEVER-IT is a large-scale dataset designed for training and evaluating fact verification systems in Italian. The dataset was derived from the English [FEVER dataset](https://aclanthology.org/N18-1074.pdf), which was published for the FEVER 2018 shared task competition. The original dataset consists of 185,445 claims manually verified against Wikipedia, annotated with labels indicating whether the evidence supports, refutes, or provides not enough information about the claim.

We based our work on an [extended version](https://huggingface.co/datasets/copenlu/fever_gold_evidence) of the FEVER dataset that, which along with several adjustments, includes synthetic evidence for the "Not Enough Info" category, enhancing its utility for training fact-checking models.

To ensure high-quality translations, our team of reviewers manually validated a subset of the data. They focused on correcting errors related to fluency, completeness, and correctness of the automatic translations compared to the original English text. This effort produced the **GOLD** dataset, a high-quality subset containing **2,063** manually validated claim-evidence pairs.

The quality of the automatic translations was evaluated by comparing the GOLD set with the corresponding portion of the SILVER set using BLEU metrics. The results indicated very high translation quality:

| Metric | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|--------|--------|--------|--------|--------|
| **Claim** | 0.9776 | 0.9695 | 0.9623 | 0.9544 |
| **Evidence** | 0.9529 | 0.9411 | 0.9309 | 0.9207 |

### Dataset Structure

The final FEVER-IT dataset is organized into three subsets:
* **Training Set**: 228,277 claim-evidence pairs (SILVER)
* **Validation Set**: 15,935 claim-evidence pairs (SILVER)
* **Test Set**: 2,063 manually validated claim-evidence pairs (GOLD)

We include the training and validation sets with claim-evidence pairs automatically translated in the SILVER dataset. For the test set, we provide only the manually validated GOLD dataset, ensuring high-quality evaluation.

Each claim is categorized into one of three classes, consistent with the original FEVER dataset:
* **Supports**
* **Refutes**
* **Not Enough Info**

The distribution in the GOLD test set is as follows:
* **Total**: 2,063
   * **Supports**: 654
   * **Refutes**: 643
   * **Not Enough Info**: 766

<!--- Note that the actual original English texts are not included in this dataset, although alignment to the source data is maintained and cross-lingual research is possible. -->

### Methodology for Using FEVER-IT

The FEVER-IT dataset is intended to facilitate the development of fact verification systems in Italian. Researchers can train models on the SILVER dataset (training and validation sets) and evaluate them using the high-quality GOLD test set.

In our experiments, we fine-tuned a model on the FEVER-IT dataset. We achieved excellent performance, with metrics of recall, precision, accuracy, and F1-score comparable to a model trained on English data. This demonstrates the dataset's effectiveness in supporting the development of robust fact verification systems in Italian.

**Note**: The focus of FEVER-IT is on claim verification using provided evidence. The evidence retrieval component is not addressed in this dataset, as the primary goal is to provide a relevant and high-quality resource for fact-checking in Italian.

### Download Dataset

To download the Fever-it dataset, please refer to [this folder](https://github.com/crux82/FEVER-it)


## Adapters Llama3 finetuned on FEVER and FEVER-IT
In the following section, you can find the models already finetuned with various modes. More detail about prompts in [this section](#prompts-in-italian) or "Prompting Engineering" Appendix in the paper <!--- XXXXXXXXXXXXXXXXX -->
| LANGUAGE | DATASET | PROMPT |Document | Download |
|:----:| :----:| :--------------: | :--: | :---------: |
|ENG|FEVER| 0-shot | No | [🤗](https://huggingface.co/sag-uniroma2/llama3_adapter_ENG_pr1_add0__split0_ep1_lr0.0001_fever-eng) &nbsp;&nbsp; |
|ENG|FEVER| 0-shot| Yes | [🤗](https://huggingface.co/sag-uniroma2/llama3_adapter_ENG_pr1_add1__split0_ep1_lr0.0001_fever-eng) &nbsp;&nbsp;|
|ENG|FEVER| 1-shot | No | [🤗](https://huggingface.co/sag-uniroma2/llama3_adapter_ENG_pr2_add0__split0_ep1_lr0.0001_fever-eng) &nbsp;&nbsp; |
|ENG|FEVER| 1-shot| Yes | [🤗](https://huggingface.co/sag-uniroma2/llama3_adapter_ENG_pr2_add1__split0_ep1_lr0.0001_fever-eng) &nbsp;&nbsp; |
|ITA|FEVER-IT| 0-shot | No | [🤗](https://huggingface.co/sag-uniroma2/llama3_adapter_ITA_pr1_add0__split0_ep1_lr0.0001_fever-it) &nbsp;&nbsp; |
|ITA|FEVER-IT| 0-shot| Yes | [🤗](https://huggingface.co/sag-uniroma2/llama3_adapter_ITA_pr1_add1__split0_ep1_lr0.0001_fever-it) &nbsp;&nbsp; |
|ITA|FEVER-IT| 1-shot | No | [🤗](https://huggingface.co/sag-uniroma2/llama3_adapter_ITA_pr2_add0__split0_ep1_lr0.0001_fever-it) &nbsp;&nbsp;  |
|ITA|FEVER-IT| 1-shot| Yes | [🤗](https://huggingface.co/sag-uniroma2/llama3_adapter_ITA_pr2_add1__split0_ep1_lr0.0001_fever-it) &nbsp;&nbsp; |


## Fine-Tuning Llama3 on FEVER and FEVER-IT

### Prerequisites
- Anaconda or Miniconda installed on your system
- Python 3.12
- Git (optional, for cloning the repository)

### Installation
1. Clone the Repository
   ```bash
   git clone https://github.com/crux82/FEVER-it.git
   cd FEVER-it
   ```

2. Create a Conda Environment
   ```bash
   conda env create -f fevertrain.yml
   conda activate fevertrain
   ```
### Fine-tuning
It is important to verify the correct row in the following instruction based on the type of fine-tuning you want to perform. Ensure you select the corresponding command for the correct setup.

English 0-shot, No Document
   ```bash
python -u train.py --lang ENG --prompt_number 1 --learn_rate 0.0001 --epochs 1 --base_model meta-llama/Meta-Llama-3-8B-Instruct --conf config.json --adddoc 0 --splitevidence 0
  ```
English 0-shot, With Document
   ```bash
python -u train.py --lang ENG --prompt_number 1 --learn_rate 0.0001 --epochs 1 --base_model meta-llama/Meta-Llama-3-8B-Instruct --conf config.json --adddoc 1 --splitevidence 0
  ```
English 1-shot, No Document
   ```bash
python -u train.py --lang ENG --prompt_number 2 --learn_rate 0.0001 --epochs 1 --base_model meta-llama/Meta-Llama-3-8B-Instruct --conf config.json --adddoc 0 --splitevidence 0
  ```
English 1-shot, With Document
   ```bash
python -u train.py --lang ENG --prompt_number 2 --learn_rate 0.0001 --epochs 1 --base_model meta-llama/Meta-Llama-3-8B-Instruct --conf config.json --adddoc 1 --splitevidence 0
  ```


Italian 0-shot, No Document
   ```bash
python -u train.py --lang ITA --prompt_number 1 --learn_rate 0.0001 --epochs 1 --base_model meta-llama/Meta-Llama-3-8B-Instruct --conf config.json --adddoc 0 --splitevidence 0
  ```
Italian 0-shot, With Document
   ```bash
python -u train.py --lang ITA --prompt_number 1 --learn_rate 0.0001 --epochs 1 --base_model meta-llama/Meta-Llama-3-8B-Instruct --conf config.json --adddoc 1 --splitevidence 0
  ```
Italian 1-shot, No Document
   ```bash
python -u train.py --lang ITA --prompt_number 2 --learn_rate 0.0001 --epochs 1 --base_model meta-llama/Meta-Llama-3-8B-Instruct --conf config.json --adddoc 0 --splitevidence 0
  ```
Italian 1-shot, With Document
   ```bash
python -u train.py --lang ITA --prompt_number 2 --learn_rate 0.0001 --epochs 1 --base_model meta-llama/Meta-Llama-3-8B-Instruct --conf config.json --adddoc 1 --splitevidence 0
  ```


 
## Test Llama3 on FEVER and FEVER-IT
### Installation
1. Clone the Repository
   ```bash
   git clone https://github.com/crux82/FEVER-it.git
   cd FEVER-it
   ```

2. Create a Conda Environment
   ```bash
   conda env create -f fevertest.yml
   conda activate fevertest
   ```
**Baseline**

English 0-shot, No Document
   ```bash
   python -u test_baseline.py --lang ENG --prompt_number 1 --adddoc 0 --splitevidence 0 --adapter "NO" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
  ```
English 0-shot, With Document
   ```bash
  python -u test_baseline.py --lang ENG --prompt_number 1 --adddoc 1 --splitevidence 0 --adapter "NO" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
```
English 1-shot, No Document
   ```bash
  python -u test_baseline.py --lang ENG --prompt_number 2 --adddoc 0 --splitevidence 0 --adapter "NO" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
```
English 1-shot, With Document
   ```bash
  python -u test_baseline.py --lang ENG --prompt_number 2 --adddoc 1 --splitevidence 0 --adapter "NO" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
```


Italian 0-shot, No Document
   ```bash
  python -u test_baseline.py --lang ITA --prompt_number 1 --adddoc 0 --splitevidence 0 --adapter "NO" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
  ```
Italian 0-shot, With Document
   ```bash
python -u test_baseline.py --lang ITA --prompt_number 1 --adddoc 1 --splitevidence 0 --adapter "NO" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
  ```
Italian 1-shot, No Document
   ```bash
python -u test_baseline.py --lang ITA --prompt_number 2 --adddoc 0 --splitevidence 0 --adapter "NO" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
  ```
Italian 1-shot, With Document
   ```bash
python -u test_baseline.py --lang ITA --prompt_number 2 --adddoc 1 --splitevidence 0 --adapter "NO" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
```
**Test finetuned models**

English 0-shot, No Document
   ```bash
   python -u test.py --lang ENG --prompt_number 1 --adddoc 0 --splitevidence 0 --adapter "sag-uniroma2/llama3_adapter_ENG_pr1_add0__split0_ep1_lr0.0001_fever-eng" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
  ```
English 0-shot, With Document
   ```bash
  python -u test.py --lang ENG --prompt_number 1 --adddoc 1 --splitevidence 0 --adapter "sag-uniroma2/llama3_adapter_ENG_pr1_add1__split0_ep1_lr0.0001_fever-eng" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
```
English 1-shot, No Document
   ```bash
  python -u test.py --lang ENG --prompt_number 2 --adddoc 0 --splitevidence 0 --adapter "sag-uniroma2/llama3_adapter_ENG_pr2_add0__split0_ep1_lr0.0001_fever-eng" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
```
English 1-shot, With Document
   ```bash
  python -u test.py --lang ENG --prompt_number 2 --adddoc 1 --splitevidence 0 --adapter "sag-uniroma2/llama3_adapter_ENG_pr2_add1__split0_ep1_lr0.0001_fever-eng" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
```


Italian 0-shot, No Document
   ```bash
  python -u test.py --lang ITA --prompt_number 1 --adddoc 0 --splitevidence 0 --adapter "sag-uniroma2/llama3_adapter_ITA_pr1_add0__split0_ep1_lr0.0001_fever-it" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
  ```
Italian 0-shot, With Document
   ```bash
python -u test.py --lang ITA --prompt_number 1 --adddoc 1 --splitevidence 0 --adapter "sag-uniroma2/llama3_adapter_ITA_pr1_add1__split0_ep1_lr0.0001_fever-it" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
  ```
Italian 1-shot, No Document
   ```bash
python -u test.py --lang ITA --prompt_number 2 --adddoc 0 --splitevidence 0 --adapter "sag-uniroma2/llama3_adapter_ITA_pr2_add0__split0_ep1_lr0.0001_fever-it" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
  ```
Italian 1-shot, With Document
   ```bash
python -u test.py --lang ITA --prompt_number 2 --adddoc 1 --splitevidence 0 --adapter "sag-uniroma2/llama3_adapter_ITA_pr2_add1__split0_ep1_lr0.0001_fever-it" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
```


## Prompts in Italian

### 0-shot Setting
The following prompt is used for 0-shot learning, where the task and classes are presented without additional information.

```plaintext
### Istruzioni
Valuta se l'affermazione è supportata dalle prove fornite. Le definizioni dei termini chiave utilizzati in questo compito sono:
- Affermazione: Una dichiarazione o asserzione sotto esame.
- Prova: Informazioni che supportano o contraddicono l'affermazione.

Rispondi con uno dei seguenti giudizi basati sulle prove fornite:
- SUPPORTS: se le prove confermano l'affermazione.
- REFUTES: se le prove contraddicono direttamente l'affermazione.
- NOT ENOUGH INFO: se le prove non sono sufficienti per determinare la validità dell'affermazione.
### Input
- Affermazione: [CLAIM HERE]
- Prova: [EVIDENCE HERE]
### Risposta: [ANSWER HERE]
```

### 1-shot Setting
The following prompt is used for 1-shot learning, where the task and classes are explained, and one example per class is provided. Notice that only the evidence is reported without the title of the original document.
```plaintext
### Istruzioni
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
### Input
- Affermazione: [CLAIM HERE]
- Prova: [EVIDENCE HERE]
### Risposta: [ANSWER HERE]
```

### 0-shot Setting with Document Title
The following prompt is used for 0-shot learning, where the task and classes are explained without additional information. Each input evidence is provided with the title of its original document.
```plaintext
### Istruzioni
Valuta se l'affermazione è supportata dalle prove fornite. Le definizioni dei termini chiave utilizzati in questo compito sono:
- Affermazione: Una dichiarazione o asserzione sotto esame.
- Prova: Informazioni che supportano o contraddicono l'affermazione.
- Documento: indica la fonte da cui è stata estratta la prova.

Rispondi con uno dei seguenti giudizi basati sulle prove fornite:
- SUPPORTS: se le prove confermano l'affermazione.
- REFUTES: se le prove contraddicono direttamente l'affermazione.
- NOT ENOUGH INFO: se le prove non sono sufficienti per determinare la validità dell'affermazione.
### Input
- Affermazione: [CLAIM HERE]
- Prova: [EVIDENCE HERE]
- Documento: [DOCUMENT HERE]
### Risposta: [ANSWER HERE]
```

### 1-shot Setting with Document Title
The following prompt is used for 1-shot learning, where the task and classes are explained, and one example per class is provided. Each input evidence is provided with the title of its original document.
```plaintext
### Istruzioni
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
### Input
- Affermazione: [CLAIM HERE]
- Prova: [EVIDENCE HERE]
- Documento: [DOCUMENT HERE]
### Risposta: [ANSWER HERE]
```

## How to cite FEVER-it

This dataset was introduced in the work *"Leveraging Large Language Models for Fact Verification in Italian"* <!--- available at the following [XXXXXXXXXXXXXXXX](XXX.pdf). -->
If you find FEVER-it useful for your research, please cite the following paper:

~~~~

~~~~

## References
Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit, FEVER: a Large-scale Dataset for Fact Extraction and VERification, NAACL-HLT 2018 [Link](https://fever.ai/dataset/fever.html)

Introducing Meta Llama 3: The most capable openly available LLM to date [Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)

Atanasova, Pepa  and Wright, Dustin  and Augenstein, Isabelle, Generating Label Cohesive and Well-Formed Adversarial Claims [dataset](https://huggingface.co/datasets/copenlu/fever_gold_evidence) [paper](https://aclanthology.org/2020.emnlp-main.256/)

## Contacts

For any questions or suggestions, you can send an e-mail to <croce@info.uniroma2.it>
