# FEVER-it

## DATASET



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

| CODE | PROMPT	| DOC	|
| --------- | :---------: | :-----: |
|	EN1	|   0-shot | NO |
|	EN2 | 0-shot |YES |
|	EN3 | 1-shot | NO  |
|	EN4 | 1-shot | YES |


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
