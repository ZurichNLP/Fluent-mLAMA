# Fluent-mLAMA
Repository with code and data for the ["Measuring the Effect of Disfluency in Multilingual Knowledge Probing Benchmarks" paper](https://arxiv.org/pdf/2510.15115), presented at EMNLP-2025.

You can also find a [poster](poster.pdf) and a short [video presentation](https://www.youtube.com/watch?v=i_ceK35VS1g) of this work. 

## 0. Pre-requisites

1. Download the initial MLAMA benchmark from the [Github repo](https://github.com/norakassner/mlama). 
2. To reproduce the ChatGPT few-shot prompt generation, you would need to generate an OpenAI token. You can generate it [here](https://platform.openai.com/api-keys), and paste it into the [`openai-api-key.txt`](openai-api-key.txt) document. 

## 1. Benchmark Modification

The facts in the MLAMA benchmark [Kassner et al., 2021](https://aclanthology.org/2021.eacl-main.284/) are translated in the templated manner: for example, to translate `R=[X] died in [Y]` with `X=Sofya Kovalevskaya` and `Y=Stockholm`, the relation `R` is translated with Google Translate once and for all X-Y pairs, and the X-Y translations are retrieved from Wikidata. This leads to ungrammaticalities or wrong translations of the templates (because NMT system has not seen the named entites within the sentence): for example, for the Russian it will end up with `Софья Ковалевская умер в Стокгольм`, where the verb `умер` is used in the wrong gender and the noun `Стокгольм` is used in the wrong case.

To address this, we modified the subpart of this benchmark with a simple trick: first, filling in the X-Y pairs in the initial English prompts, and then translating **full sentences** with Google Translate (and optionally ChatGPT).

Codes and data: 
* [`modification.ipynb`](modification.ipynb) - Jupyter notebook with step-by-step generation of translations and their preparation for MLAMA evaluation.
* [`datamodifier.py`](datamodifier.py) - source code for the notebook.
* [`fixed_data`](fixed_data) - results of the code for 9 languages and subset of relations covered in the paper (unzip into the folder locally).
* [`chatgpt_prompts`](chatgpt_prompts) - few-shot prompts needed to generate the ChatGPT translations (for Experiment 1 with Slavic languages).
* [`reader.py`](reader.py) - code for mLAMA reader ([borrowed](https://github.com/norakassner/mlama/blob/master/dataset/reader.py) from the initial MLAMA dataset)

## 2. LLM Testing

Now we want to test our hypothesis: does prompting an LLM with fluent sentences help retrieve more facts compared to the disfluent templated translations? 

We are doing it in a following manner: 
- for each fact (e.g. `Sofya Kovalevskaya died in Stockholm`) we have a templated translation and a full-sentence translation. 
- in the dataset ([part 1](#1-benchmark-modification)), we already split the "prompt base" (`Sofya Kovalevskaya died in`) from an object of interest (`Stockholm`).
- we take a prompt base, concatenate it with the correct object and with a set of incorrect objects ("distractors", e.g. `Moscow, London, Paris`), and feed it to an LLM (in our case, it's [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)).
- we get the log-probability scores of each object (both correct and distractor ones), and rank them by log-probabilities. If the model has ranked the correct objects high enough (for example, in top-3 options), we say that it **knows** a fact. 

Codes and data:
* [`processor.py`](processor.py) - module used for prompting an LLM.
* [`script.py`](script.py) - python script to run evaluation (for a given language and a range of relations).
* [`stats`](stats) - resulting data with the ranks and log-probabilities of the objects
  * * [`stats_wide`](stats_wide) - distributions for the mode where aliases are counted as correct objects

## 3. Evaluation, Graphs and Tables

