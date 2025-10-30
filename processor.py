import random
from collections.abc import Iterable
import pandas as pd
import hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from huggingface_hub import login


class Processor:
    """
    Processor class is designed to evaluate a given LLMs on multilingual factual dataset,
    using various prompt translation techniques to compare performance metrics and manage results.

    Processor assists in model evaluation by analyzing performance across different prompt translation methods:
            - templated translation - default translation in several datasets including MLAMA benchmark.
            - Google Translate translation (GT) - using the full sentence translation by Google.
            - ChatGPT translation (ChatGPT) - using ChatGPT few-shot constrained translation.

    """


    def __init__(self, datapath='./fixed_data/', model="meta-llama/Llama-2-7b-chat-hf", twomodes=False):
        """
        Initializes a class instance with specified model, data path, and mode settings (experiment 1 or 2).

        :param datapath (str): The file path to the directory with fixed mLAMA translations. Default is './fixed_data/'.
        :param model (str): The Huggingface model name. Default is "meta-llama/Llama-2-7b-chat-hf".
        :param twomodes (bool): Whether template+Google Translate is compared (True) or ChatGPT as well (False).
            Default is True.

        Attributes:
            model: The loaded huggingface model instance based on the specified model name (so far only LLaMA-2-7b is implemented).
            tokenizer: The tokenizer associated with the loaded model.


        """
        self.datapath = datapath
        self.model2source = {
            "meta-llama/Llama-2-7b-chat-hf": "hf", "bigscience/bloom-7b1": "hf",
            "Tower-Babel/Babel-9B": 'hf'
        }
        self.model, self.tokenizer = self._load_model(model)
        self.twomodes = twomodes

    def single_processing_run(self, lang, rel, dataset_spec=None, trial_run=False, mode='narrow', topk=3):
        """

        Makes evaluation of a given language x relation subset to evaluate the performance of the given LLM (self.model)
        under different types of mutlilingual prompt translations:
        template, Google Translate, and optionally, a ChatGPT-based translation.
        It saves the computed scores (ranks of the correct objects and their probabilities)
         into a specified directory and returns the data for further use.

        Args:
        :param lang (str): MLAMA language code for the data to be processed.
        :param rel (str): The Wikidata relation ID to be processed.
        :param dataset_spec (str, optional): Specification of whether subset has:
               - two translations (template, Google Translate), in that case equals "nogpt"
               - or three translations (template, Google Translate, ChatGPT), in that case equals "chatgpt"
        :param trial_run (bool, optional): Whether to perform a trial run on top 10 data points.
        :param mode (str): Specifying if we allow object aliases as correct anwser options:
                            - 'narrow' (default) for not including,
                            - 'wide' for including.
        :param topk (int): The top k value of the Recall@k metric. Default is 3.

        Returns:
        :return: langrel_df, DataFrame containing metric scores for the given language-relation pair.
        """

        langrel_df = self._load_langrel(lang, rel, dataset_spec)

        if trial_run:
            langrel_df = langrel_df[:10]
        adversarial_pool = self._prepare_adversarial_pool(langrel_df)
        target_dir = 'stats' if mode == 'narrow' else 'stats_wide'
        langrel_df['template_true_ranks'], langrel_df['template_top1_correct'], langrel_df['template_prompt_prob'], \
        langrel_df[f'template_top{topk}_correct'] = zip(*langrel_df.apply(
            lambda row: self._get_single_prompt_scores('template_prompt', self.tokenizer, self.model, row,
                                                       adversarial_pool, mode=mode, topk=topk), axis=1))

        langrel_df['gt_true_ranks'], langrel_df['gt_top1_correct'], langrel_df['gt_prompt_prob'], langrel_df[
            f'gt_top{topk}_correct'] = zip(*langrel_df.apply(
            lambda row: self._get_single_prompt_scores('gt_prompt', self.tokenizer, self.model, row, adversarial_pool,
                                                       mode=mode, topk=topk), axis=1))
        # TODO: generate saving to file as a parameter
        if self.twomodes:
            langrel_df.to_csv(f'{target_dir}/{lang}_{rel}_template_gt.tsv', sep='\t')
        else:
            langrel_df['chatgpt_true_ranks'], langrel_df['chatgpt_top1_correct'], langrel_df['chatgpt_prompt_prob'], langrel_df[f'chatgpt_top{topk}_correct'] = zip(*langrel_df.apply(lambda row: self._get_single_prompt_scores('chatgpt_prompt', self.tokenizer, self.model, row, adversarial_pool, mode=mode, topk=topk), axis=1))
            langrel_df.to_csv(f'{target_dir}/{lang}_{rel}_template_gt_chatgpt.tsv', sep='\t')
        return langrel_df


    def _load_langrel(self, lang, rel, dataset_spec, multiling=False):
        """
        _load_langrel loads a language-relation dataset, prepares alias strings, and filters invalid entries.

        Parameters:
        :param lang (str): MLAMA Language code.
        :param rel (str): Wikidata Relation ID.
        :param dataset_spec (str, optional): Specification of whether subset has:
               - two translations (template, Google Translate), in that case equals "nogpt"
               - or three translations (template, Google Translate, ChatGPT), in that case equals "chatgpt"
        :param multiling (bool): not used now; intended for multilingual correct object analysis.

        :return: langrel_df, DataFrame containing input data for the given language-relation pair.
        """
        spec = '_' + dataset_spec if dataset_spec is not None else '_default'
        mult = '_multiling' if multiling else ''
        langrel_fname = f'{lang}_{rel}{spec}{mult}.tsv'
        langrel_df = pd.read_csv(self.datapath + langrel_fname, sep='\t')
        langrel_df['aliases'] = langrel_df['aliases'].apply(lambda x: x.split('|||'))
        # print(langrel_df.shape)
        langrel_df = langrel_df[(~langrel_df['gt_infl_object'].eq('ERROR')) & ((~langrel_df['gt_prompt'].isna())) & (
        (langrel_df['gt_prompt'] != ''))]

        return langrel_df

    def _load_model(self, model):
        """
        Loads a pre-trained model and tokenizer from Huggingface hub. Currently only LLaMA-2-7b is supported.

        :param model (str): The name or identifier of the model to load, determining its
            source type and the initialization procedures to apply.

        Returns:
        :return: model object
        :return: tokenizer object
        """
        if self.model2source[model] == 'hf':
            if 'llama' in model:
                with open('./llama-mlama-tok.txt', 'r', encoding='utf-8') as f:
                    self.hf_token = f.read()
                login(token=self.hf_token)
            tokenizer = AutoTokenizer.from_pretrained(model)
            model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")
        else:
            # TODO: add interface for other types of models
            pass
        return model, tokenizer

    def _prepare_adversarial_pool(self, df, mode='general'):
        """
        Prepares pool of adversarial object names ("distractors" in paper) by flattening the input data.

        :param df (pd.DataFrame): Input dataframe.
        :param mode (str): currently unused, defaults to "general".

        :return: adv_pool, set of distractors for a given object.
        """
        def flatten(xs):
            for x in xs:
                if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                    yield from flatten(x)
                else:
                    yield x

        if mode == 'general':
            # main experiment mode
            if self.twomodes:
                adv_pool = df[['obj_label', 'gt_infl_object', 'aliases']].values.tolist()
            else:
                adv_pool = df[['obj_label', 'gt_infl_object', 'chatgpt_infl_object', 'aliases']].values.tolist()
            adv_pool = set([adv for adv in flatten(adv_pool)])
            adv_pool = set([adv for adv in adv_pool if type(adv) == str and len(adv) > 0])
        elif mode == 'multiling':
            # outline for next experiments: selecting adversaries from other languages as well
            pass
        return adv_pool

    def _prepare_completions(self, line, adversarial_pool, mode='narrow', adv_size=50):
        """

        Prepares the set of prompts that would be sent to an LLM for a single fact.
        The evaluation procedure is to prompt the model with a set of sentences
        "London is a capital of {the UK, France, China...}", and rank them by the probablities of the objects (in curly brackets).
        To do that, we need to generate a set of sentences "London is a capital of the UK", "London is a capital of France", etc.,
        This is the first step of this preparation: selecting the pool of
        correct objects (depends if we include aliases or not,)
        and adversarial objects (called "distractors" in paper).


        :param line (dict): A single data-point containing object form,
                            Google Translate object form, aliases, and optionally ChatGPT's object form.
        :param adversarial_pool (set): A set of adversarial candidates (distractors).
        :param mode (str): Determines which object forms to count as correct:
                           - if 'narrow', includes only the template and Google Translate (and ChatGPT if available) forms.
                           - if 'wide', aliases are also included.
        :param adv_size (int): The maximum number of adversarial completions to sample. may be smaller if the pool size is less than adv_size.

        :return: true_completions, list of true completions derived from the input data.
        :return: adversarial_completions, a list of sampled adversarial completions.
        """

        initial_form, gt_infl_form = line['obj_label'], line['gt_infl_object']

        aliases = line['aliases']
        if mode == 'narrow':
            true_completions = [initial_form, gt_infl_form]
        elif mode == 'wide':
            true_completions = [initial_form, gt_infl_form] + aliases
        if not self.twomodes:
            chatgpt_infl_form = line['chatgpt_infl_object']
            true_completions += [chatgpt_infl_form]
        true_completions = [c for c in set(true_completions) if type(c) == str and len(c) > 0]
        #correct_set = set([initial_form, gt_infl_form] + aliases)  # , chatgpt_infl_form - TODO make sure the line below works
        correct_set = set(true_completions + aliases)
        correct_set = set([c for c in correct_set if type(c) == str and len(c) > 0])
        adversarial_pool = adversarial_pool - correct_set

        # selecting the adv_size:
        # old way - through random package, the random sample is different at every experiment run
        #        if len(adversarial_pool) < adv_size:
        #            real_adv_size = len(adversarial_pool)
        #            print(f'adversarial_pool size is less than adv_size, setting adv_size to {real_adv_size}')
        #        else:
        #            real_adv_size = adv_size
        #        adversarial_completions = random.sample(list(adversarial_pool), real_adv_size)
        # new way - through hashlib; stabilizes the random sample
        def stable_sample(lst, k):
            def hash_score(item):
                h = hashlib.sha256(str(item).encode()).hexdigest()
                score = int(h, 16)

                return score

            sorted_lst = sorted(lst, key=hash_score)

            return sorted_lst[:k]

        if len(adversarial_pool) < adv_size:
            print(f'adversarial_pool size is smaller than adv_size, setting adv_size to {len(adversarial_pool)}')
            adv_size = len(adversarial_pool)
        adversarial_completions = stable_sample(list(adversarial_pool), adv_size)

        return true_completions, adversarial_completions


    def _get_single_prompt_scores(self, checked_col, tokenizer, model, line, adversarial_completions, mode='narrow',
                                  topk=1):
        """
        Runs the correct and distractor prompts through LLM and compares the probabilities of correct completions
        and adversarial completions.

        Args:
        :param checked_col (str): The column index for line used as the base prompt
                                  ('template_prompt', 'gt_prompt' or 'chatgpt_prompt').
        :param tokenizer (HF tokenizer): tokenizes prompts and completions.
        :param model (HF model): The LLM being tested.
        :param line (dict): a data point containing the prompt, correct object, aliases, etc.
        :param adversarial_completions (set): total pool of adversarial/distractor objects for a given object.
        :param mode (str): The mode for processing completions (default: 'narrow').
        :param topk (int): The top k value of the Recall@k metric. Default is 1.


        :return true_completions_ranks (dict): Ranks of true completions within the sorted probabilities.
        :return is_top1_correct (int): Indicator if the top-ranked completion is correct (1 if true, 0 otherwise).
        :return prompt_prob (float): Log-probability of the base prompt (London is a capital of...).
        :return is_topk_correct (int): Indicator if any of the top-k ranked completions are correct (1 if true, 0 otherwise).
        """

        # create the pool of correct object and distractors
        true_completions, adversarial_completions = self._prepare_completions(line, adversarial_completions, mode=mode)
        # take the base of the prompt ("London is a capital of...")
        prompt = line[checked_col]
        #print(prompt)
        # run the LLM with the correct prompts ("London is a capital of..." + ["Great Britain", "UK"])
        true_completions_probs, prompt_prob = self._get_sequence_probability(prompt, tokenizer, model, true_completions)
        # run the LLM with the distractors ("London is a capital of..." + ["Russia", "USA"])
        adversarial_completions_probs, _ = self._get_sequence_probability(prompt, tokenizer, model,
                                                                          adversarial_completions)

        # sort the correct and distractor prompts by the probabilities of the objects (given base of the prompts)
        total_probs = true_completions_probs | adversarial_completions_probs

        sorted_probs = dict(sorted(total_probs.items(), key=lambda item: item[1], reverse=True))

        true_completions_ranks = {}
        is_top1_correct = 0
        for idx, key in enumerate(sorted_probs.keys()):
            if key in true_completions:
                true_completions_ranks[key] = idx + 1
                if idx < 1:
                    is_top1_correct = 1

        #print('is_top1_correct: ', str(is_top1_correct))
        if topk > 1:
            is_topk_correct = 0
            for idx, key in enumerate(sorted_probs.keys()):
                if key in true_completions:
                    true_completions_ranks[key] = idx + 1
                    if idx < topk:
                        is_topk_correct = 1
        else:
            is_topk_correct = is_top1_correct
        #print('is_topk_correct: ', str(is_topk_correct))

        return true_completions_ranks, is_top1_correct, prompt_prob, is_topk_correct


    def _get_sequence_probability(self, prompt, tokenizer, model, completions):
        """
        Calculates the log probabilities of the prompt base ("London is a capital of...")
         and given text completions (["UK", "Great Britain", "France"]) for a given LLM model.

        IMPORTANT: this function was generated with a help of ChatGPT and then post-checked by the author.

        :param prompt (str): The prompt base ("London is a capital of...")
        :param tokenizer (HF tokenizer): tokenizes prompts and completions.
        :param model (HF model): The LLM being tested.
        # TODO - change to self.model, self.tokenizer? But then would have to initiate differently for every model
        :param completions (List[str]): candidate object completions ["UK", "Great Britain", "France"] to evaluate probabilities.

        :return completion_probs (Dict[str, float]): mapping each object to its log probability given the prompt.
        :return prompt_prob (float): The log probability of the prompt base.

        """
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        # Dictionary to store probabilities for each completion
        completion_probs = {}

        for completion in completions:
            # print(completion)
            completion_ids = tokenizer(completion, add_special_tokens=False, return_tensors="pt").input_ids.to(
                model.device)
            full_input_ids = torch.cat([input_ids, completion_ids], dim=-1)
            #    print(completion_ids)
            #    print(full_input_ids)
            with torch.no_grad():
                outputs = model(full_input_ids)
                logits = outputs.logits

            # Calculate log probabilities for each token in the completion
            log_prob = 0
            #    print(completion_ids.shape)
            for i in range(completion_ids.shape[1]):
                token_logits = logits[0, -(completion_ids.shape[1] - i) - 1, :]
                # print(token_logits.shape)
                token_prob = F.softmax(token_logits, dim=-1)[completion_ids[0, i]]
                #        print(token_prob)
                log_prob += torch.log(token_prob)

            completion_probs[
                completion] = log_prob.item()  # torch.exp(log_prob).item()  # Convert log-prob to probability

        # Calculate the probability of the initial prompt
        with torch.no_grad():
            prompt_outputs = model(input_ids)
            prompt_logits = prompt_outputs.logits

        prompt_log_prob = 0
        for i in range(1, input_ids.shape[1]):  # Start from 1 since first token has no previous context
            token_logits = prompt_logits[0, i - 1, :]
            token_prob = F.softmax(token_logits, dim=-1)[input_ids[0, i]]
            prompt_log_prob += torch.log(token_prob)

        prompt_prob = prompt_log_prob.item()  # torch.exp(prompt_log_prob).item()  # Convert log-prob to probability
        return completion_probs, prompt_prob