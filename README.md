# FEVER-it

## DATASET
FEVER-IT is a large-scale dataset designed for training and evaluating fact verification systems in Italian. It is derived from the original FEVER dataset, an extensive resource for fact-checking in English, which contains claims and their corresponding evidence from Wikipedia **[CITARE]**.

The entire FEVER dataset was translated into Italian using **MADLAD-400**, a multilingual translation model based on the Transformer architecture. This automatic translation process generated the **SILVER** dataset, comprising **246,275** claim-evidence pairs.

Multiple reviewers manually validated a subset of the data to ensure high-quality translations. They focused on correcting errors related to fluency, completeness, and correctness of the automatic translations compared to the original English text. This effort produced the **GOLD** dataset, a high-quality subset containing **2,063** manually validated claim-evidence pairs.

The quality of the automatic translations was evaluated by comparing the GOLD set with the corresponding portion of the SILVER set using BLEU metrics. The results indicated very high translation quality:

| Metric | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|--------|--------|--------|--------|--------|
| **Claim** | 0.9776 | 0.9695 | 0.9623 | 0.9544 |
| **Evidence** | 0.9529 | 0.9411 | 0.9309 | 0.9207 |

### Dataset Structure

The SILVER dataset is divided into three sets:
* **Training Set**: 228,277 claim-evidence pairs
* **Validation Set**: 15,935 claim-evidence pairs
* **Test Set**: 2,063 manually validated claim-evidence pairs

Each claim is categorized into one of three classes, consistent with the original FEVER dataset:
* **Supports**
* **Refutes**
* **Not Enough Info**

The distribution in the GOLD test set is as follows:
* **Total**: 2,063
   * **Supports**: 654
   * **Refutes**: 643
   * **Not Enough Info**: 766

The dataset includes only the translated claims and evidence, along with references to the document sentence ID of the evidence, and the label provided in the original dataset. While it does not include the original English texts, it maintains alignment with the source data for potential cross-lingual research.

### Methodology for Using FEVER-IT

The FEVER-IT dataset is intended to facilitate the development of fact verification systems in Italian. Researchers can train models on the SILVER dataset (training and validation sets) and evaluate them using the high-quality GOLD test set.

In our experiments, we fine-tuned a model on the FEVER-IT dataset. We achieved excellent performance, with metrics of recall, precision, accuracy, and F1-score comparable to a model trained on English data **[CITARE E MOSTRARE MIGLIORI RISULTATI RISPETTO AL MODELLO ADDESTRATO IN INGLESE]**. This demonstrates the dataset's effectiveness in supporting the development of robust fact verification systems in Italian.

**Note**: The focus of FEVER-IT is on claim verification using provided evidence. The evidence retrieval component is not addressed in this dataset, as the primary goal is to provide a relevant and high-quality resource for fact-checking in Italian.

### References

For comprehensive information on the original FEVER dataset and its extensions, please refer to the **[Original FEVER Dataset Citation]**.

Our dataset is derived from an extended version of FEVER that includes evidence for the "Not Enough Info" category, which is particularly useful for training fact-checking models. Specifically, we utilized the dataset provided in **[Citation B]**. In our version, we provide only the manually validated test set, focusing on the claim-evidence pairs with higher accuracy rather than the entire test set. This approach ensures that evaluations are based on high-quality, accurate data.

In contrast to the reference dataset, where each claim could be associated with multiple pieces of evidence (corresponding to multiple lines from Wikipedia articles), we have separated these into individual claim-evidence pairs. Consequently, the number of rows in our dataset is higher than in the original. This granular structure enhances the precision and effectiveness of training and evaluating fact verification models.

## ADAPTERS LLAMA3 FINETUNED ON FEVER AND FEVER-IT
In the following section, you can find the models already finetuned with various modes.
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


## FINE-TUNING LLAMA3 ON FEVER AND FEVER-IT

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


 
## TEST LLAMA3 ON FEVER AND FEVER-IT
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
**BASELINE**

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
**TEST FINETUNED MODEL**

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
