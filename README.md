# FEVER-it

## DATASET


## ADAPTERS LLAMA3 FINETUNED ON  
In the following section, you can find the models already finetuned with various modes.
| LANGUAGE | PROMPT |Document | Download |
|:----:| :--------------: | :--: | :---------: |
|ENG| 0-shot | No | [🤗](https://huggingface.co/sag-uniroma2) &nbsp;&nbsp; |
|ENG| 0-shot| Yes | [🤗](https://huggingface.co/sag-uniroma2) &nbsp;&nbsp;|
|ENG| 1-shot | No | [🤗](https://huggingface.co/sag-uniroma2) &nbsp;&nbsp; |
|ENG| 1-shot| Yes | [🤗](https://huggingface.co/sag-uniroma2) &nbsp;&nbsp; |
|ITA| 0-shot | No | [🤗](https://huggingface.co/sag-uniroma2) &nbsp;&nbsp; |
|ITA| 0-shot| Yes | [🤗](https://huggingface.co/sag-uniroma2) &nbsp;&nbsp; |
|ITA| 1-shot | No | [🤗](https://huggingface.co/sag-uniroma2) &nbsp;&nbsp;  |
|ITA| 1-shot| Yes | [🤗](https://huggingface.co/sag-uniroma2) &nbsp;&nbsp; |


## FINE-TUNING LLAMA3 ON FEVER-IT

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
### FINE TUNING
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


 
## TEST LLAMA3 ON FEVER-IT
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
**CALCULATE BASELINE**

English 0-shot, No Document
   ```bash
   python -u test_baseline.py --lang ENG --prompt_number 1 --adddoc 0 --splitevidence 0 --adapter "" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
  ```
English 0-shot, With Document
   ```bash
  python -u test_baseline.py --lang ENG --prompt_number 1 --adddoc 1 --splitevidence 0 --adapter "" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
```
English 1-shot, No Document
   ```bash
  python -u test_baseline.py --lang ENG --prompt_number 2 --adddoc 0 --splitevidence 0 --adapter "" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
```
English 1-shot, With Document
   ```bash
  python -u test_baseline.py --lang ENG --prompt_number 2 --adddoc 1 --splitevidence 0 --adapter "" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
```


Italian 0-shot, No Document
   ```bash
  python -u test_baseline.py --lang ITA --prompt_number 1 --adddoc 0 --splitevidence 0 --adapter "" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
  ```
Italian 0-shot, With Document
   ```bash
python -u test_baseline.py --lang ITA --prompt_number 1 --adddoc 1 --splitevidence 0 --adapter "" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
  ```
Italian 1-shot, No Document
   ```bash
python -u test_baseline.py --lang ITA --prompt_number 2 --adddoc 0 --splitevidence 0 --adapter "" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
  ```
Italian 1-shot, With Document
   ```bash
python -u test_baseline.py --lang ITA --prompt_number 2 --adddoc 1 --splitevidence 0 --adapter "" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
```
**TEST FINETUNED MODEL**


English 0-shot, No Document
   ```bash
   python -u test.py --lang ENG --prompt_number 1 --adddoc 0 --splitevidence 0 --adapter "" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
  ```
English 0-shot, With Document
   ```bash
  python -u test.py --lang ENG --prompt_number 1 --adddoc 1 --splitevidence 0 --adapter "" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
```
English 1-shot, No Document
   ```bash
  python -u test.py --lang ENG --prompt_number 2 --adddoc 0 --splitevidence 0 --adapter "" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
```
English 1-shot, With Document
   ```bash
  python -u test.py --lang ENG --prompt_number 2 --adddoc 1 --splitevidence 0 --adapter "" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
```


Italian 0-shot, No Document
   ```bash
  python -u test.py --lang ITA --prompt_number 1 --adddoc 0 --splitevidence 0 --adapter "" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
  ```
Italian 0-shot, With Document
   ```bash
python -u test.py --lang ITA --prompt_number 1 --adddoc 1 --splitevidence 0 --adapter "" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
  ```
Italian 1-shot, No Document
   ```bash
python -u test.py --lang ITA --prompt_number 2 --adddoc 0 --splitevidence 0 --adapter "" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
  ```
Italian 1-shot, With Document
   ```bash
python -u test.py --lang ITA --prompt_number 2 --adddoc 1 --splitevidence 0 --adapter "" --base_model "meta-llama/Meta-Llama-3-8B-Instruct"
```
