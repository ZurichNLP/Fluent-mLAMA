# Fluent-mLAMA
Repository with code and data for the ["Measuring the Effect of Disfluency in Multilingual Knowledge Probing Benchmarks" paper](https://arxiv.org/pdf/2510.15115), presented at EMNLP-2025.

The code and the data will be presented at the conference date.

## 0. Pre-requisites

1. Download the initial MLAMA benchmark from the [Github repo](https://github.com/norakassner/mlama). 
2. To reproduce the ChatGPT few-shot prompt generation, you would need to generate an OpenAI token. You can generate it [here](https://platform.openai.com/api-keys), and paste it into the [`openai-api-key.txt`](openai-api-key.txt) document. 

## 1. Benchmark Modification

The facts in the MLAMA benchmark [Kassner et al., 2021](https://aclanthology.org/2021.eacl-main.284/) are translated in the templated manner: for example, to translate `R=[X] died in [Y]` with `X=Sofya Kovalevskaya` and `Y=Stockholm`, the relation `R` is translated with Google Translate once and for all X-Y pairs, and the X-Y translations are retrieved from Wikidata. This leads to ungrammaticalities or wrong translations of the templates (because NMT system has not seen the named entites within the sentence): for example, for the Russian it will end up with `Софья Ковалевская умер в Стокгольм`, where the verb `умер` is used in the wrong gender and the noun `Стокгольм` is used in the wrong case.

To address this, we modified the subpart of this benchmark with a simple trick: first, filling in the X-Y pairs in the initial English prompts, and then translating **full sentences** with Google Translate (and optionally ChatGPT).

Main codes and data: 
* [`modification.ipynb`](modification.ipynb) - Jupyter notebook with step-by-step generation of translations and their preparation for MLAMA evaluation.
* [`datamodifier.py`](datamodifier.py) - source code for the notebook.
* [`fixed_data`](fixed_data) - results of the code for 9 languages and subset of relations covered in the paper (unzip into the folder locally).
* [`chatgpt_prompts`](chatgpt_prompts) - few-shot prompts needed to generate the ChatGPT translations (for Experiment 1 with Slavic languages).
* [`reader.py`](reader.py) - code for mLAMA reader ([borrowed](https://github.com/norakassner/mlama/blob/master/dataset/reader.py) from the initial MLAMA dataset)