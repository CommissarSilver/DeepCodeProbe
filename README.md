`DeepCodeProbe` is a tool designed for probing small ML models trained on syntactic representations of code in order to provide interpretability on their syntax learning capbilities alongside the represeantions they learn. The tool is designed to be model agnostic and can be used with any model that uses AST/CFG as input.

# Dependencies
The experiments were carried out using Python 3.10.
Install the dependencies for DeepCodeProbe with:
```bash
pip install -r requirements.txt
```
Additionally, each of the models under study have their own dependencies. In order to train the models and replicate the results, you need to install the dependencies for each model. 

## AST-NN
First, create a virtual environment and activate it:
```bash
python -m venv astnn
source astnn/bin/activate
```
Then, install the dependencies for AST-NN:
```bash
pip install -r src/astnn/requirements.txt
```

## FuncGNN
First, create a virtual environment and activate it:
```bash
python -m venv funcgnn
source funcgnn/bin/activate
```
Then, install the dependencies for FuncGNN:
```bash
pip install -r src/funcgnn/requirements.txt
```

## SummarizationTF
First, create a virtual environment and activate it:
```bash
python -m venv summarizationtf
source summarizationtf/bin/activate
```
Then, install the dependencies for SummarizationTF:
```bash
pip install -r src/summarization_tf/requirements.txt
```

## CodeSumDRL
First, create a virtual environment and activate it:
```bash
python -m venv code_sum_drl
source code_sum_drl/bin/activate
```
Then, install the dependencies for CodeSumDRL:
```bash
pip install -r src/code_sum_drl/requirements.txt
```

# Training The Models
In order to train the models, you need to download the dataset. Each model has its own dataset. The datasets can be downloaded from the following links:
- AST-NN: [AST-NN Dataset](https://github.com/zhangj111/astnn)
- FuncGNN: [FuncGNN Dataset](https://github.com/aravi11/funcGNN)
- SummarizationTF: [SummarizationTF Dataset](https://github.com/sh1doy/summarization_tf)
- CodeSumDRL: [CodeSumDRL Dataset](https://github.com/wanyao1992/code_summarization_public/tree/master)

After downloading the datasets, put them in the `dataset` directory at the root of each models' source directory.
Afterwards, you can train the models by running the following the instructions in the README files of each model:
- AST-NN: `src/ast_nn/README.md`
- FuncGNN: `src/funcgnn/README.md`
- SummarizationTF: `src/summarization_tf/README.md`
- CodeSumDRL: `src/code_sum_drl/README.md`

# Training The Probes
After training the models, you can train the probes by running the following command:
```bash
python src/probe_model.py --model {model_name}
```
Where `{model_name}` is the name of the model you want to train the probe for. Please note that each model requires a different probe configuration. The configurations for each model are outlined in `probe_model.py`.

# Reproducing The Validation Results
After training the probes and the models, you can reproduce the validation results by running the following command:
```bash
python src/validate_probe.py --model {model_name}
```
Where `{model_name}` is the name of the model you want to evaluate the probe for. Similar to training the probes, each model requires a different probe configuration. The configurations for each model are outlined in `validate_probe.py`.

```
