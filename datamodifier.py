import pandas as pd
import re
import time

from deep_translator import GoogleTranslator
from hanziconv import HanziConv

from wikidata.client import Client
import wikipedia
from openai import OpenAI

import stanza
import textdistance

from reader import MLama

class DataModifier:

    '''
    full-sentence modification of a single (lang, relation) part of the templated MLAMA dataset.
    Gets the data from the MLAMA dataset,
    translates the full-sentence English templates with Google Translate,
    optionally adds ChatGPT-generated 20-shot translations,
    splits the full-sentence translated sentences into prompt and object parts (London is the capital of/the UK),
    adds aliases for object URIs, and saves the resulting DataFrame to a TSV file.

    Main attributes:
        df (DataFrame): Main data frame used for processing. Loaded from file or created dynamically.
        template_dict (dict): Dictionary containing unpacked data templates. Exists only if data is dynamically unpacked.
        idx2alias (dict): Dictionary mapping entity URIs to their associated alias pool.

    '''

    def __init__(self, init_path, lang, relation, trial=False, gt_intermediate_folder='gt_doc', continuation=False):
        """
        Initialize the class with required parameters and set up necessary libraries, clients, and APIs. It also loads or processes initial data
        based on the input parameters.

        :param init_path (str): Path to the MLAMA dataset folder (specifically, the ~/data/mlama1.1/ folder).
        :param lang (str): Language code (e.g., 'en' for English, 'ru' for Russian).
        :param relation (str): Wikidata relation type to process data with (e.g., 'P19' for 'was born in').
        :param gt_intermediate_folder (str): Folder intended for intermediate Google Translate translations. Defaults to 'gt_doc'.
        :param trial (bool): Flag to indicate whether to process a small subset of data for testing purposes. Defaults to False.
        :param continuation (bool): Flag to indicate whether to continue processing from an existing intermediate data file. Defaults to False.

        Attributes (in addition to params):
            mlama2google (dict): Mapping of MLAMA language codes to Google Translate codes. Currently unused.
            wikidata_client (Client): Client instance for accessing Wikidata.
            openai_client (OpenAI): OpenAI client instantiated with API credentials.
            lemmatizer (Pipeline): Stanza pipeline for language-specific lemmatization.
        """
        self.init_path = init_path
        self.lang = lang
        self.relation = relation
        self.gt_intermediate_folder = gt_intermediate_folder
        if continuation:
            self.df = pd.read_csv(f'intermediate_data/{self.lang}_{self.relation}.tsv', sep='\t', index_col='Unnamed: 0')
        else:
            self.df, self.template_dict = self._unpack_data(trial=trial)
            with open(f'chatgpt_prompts/translation20/{self.lang}_{self.relation}.txt', 'a', encoding='utf-8') as f:
                f.write('')
        # initializing the necessary libraries, clients and API
        self.mlama2google = {'en': 'en', 'ru': 'ru', 'he': 'iw', 'hr': 'hr',
                             'zh': 'zh-CN', 'es': 'es', 'cs': 'cs', 'uk': 'uk'} # not used now

        # wikidata and wikipedia
        self.wikidata_client = Client()
        # self.wikipedia_client = wikipedia.Wikipedia(language=lang) TODO: for extended info from wikipedia
        # openAI client
        with open('openai-api-key.txt', 'r', encoding='utf-8') as f:
            openai_key = f.read()
        self.openai_client = OpenAI(api_key=openai_key)
        # stanza lemmatizer
        stanza.download(lang)
        self.lemmatizer = stanza.Pipeline(lang, processors='tokenize,lemma,pos,depparse', verbose=False, use_gpu=False)

    def add_google_translations(self, mode='manual_doc', aux_path=None):
        '''
        translation of a df column with Google Translate (can be applied to different modes of translation)
        :param mode: str, currently default is 'manual_doc', which means you have to generate the Google Translation as
                    a separate tsv file and then load it here. To do that:
                    1. take the english sentences from an xlsx file from `gt_doc/src` folder (it's already saved)
                    2. translate them through the document translation in Google web interface
                    3. upload them to `gt_doc/tgt` folder
                    4. change the column name with the translations to "gt_translation"
        :param aux_path: str, path to the tsv file with the translations. Unused if mode is 'manual_doc'.
        :returns: 'gt_translation' column in the df
        '''
        # isolated translations of 1 sentence
        if mode == 'single':
            # currently not working due to poor quality of non-paid Google Translate API
            self.df['gt_translation'] = self.df['en_template'].apply(lambda x: self._translate_one_sentence(x))
        elif mode == 'old_workaround':
            with open(aux_path, 'r', encoding='utf-8') as f:
                aux_df = pd.read_csv(f, sep='\t')
                en2gt = aux_df[['en_template', 'gt_translation']].set_index('en_template')['gt_translation'].to_dict()
                self.df['gt_translation'] = self.df['en_template'].apply(lambda x: en2gt[x])
        elif mode == 'manual_doc':
            # temporary workaround while I haven't sorted translation thru API for free
            # file format: tsv, tab-separated, 2 columns: 'en_template', 'gt_translation'
            gt_df = pd.read_excel(f'{self.gt_intermediate_folder}/tgt/{self.lang}_{self.relation}.xlsx', names=['idx', 'gt_translation'])
            assert gt_df.shape[0] == self.df.shape[0]
            self.df['gt_translation'] = gt_df['gt_translation']

    def add_chatgpt_translation(self, system_prompt_folder='./chatgpt_prompts/translation20/'):
        """
        Adds a ChatGPT-generated 20-shot translation.

        :param system_prompt_folder (str): The directory where 20-shot prompts for ChatGPT are stored.
        :return: 'chatgpt_translation' column in the df
        """
        with open(system_prompt_folder + f'{self.lang}_{self.relation}.txt', 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        #print(system_prompt)
        a = time.time()
        print('creating translated+enhanced chatgpt sentences...')

        self.df['chatgpt_translation'] = self.df.apply(lambda row: self._chatgpt_prompt_one_entity({
            'sub_label': row[f'sub_label'], #{self.lang}_
            'obj_label': row[f'obj_label'],
            'en_templated_sent': row['en_template'],
        }, system_prompt, mode='translation'), axis=1)

        b = time.time()
        print(f'finished in {b - a} seconds')
        return None

    def add_aliases(self):
        """
        Adds object aliases to the DataFrame based on Wikidata URIs.

        :return: idx2alias : dict, a mapping of entity URIs to their associated alias pool.
        :return: 'aliases' column in self.df alias lists for each object URI.

        """
        self.idx2alias = self._get_alias_pool()
        self.df['aliases'] = self.df['obj_uri'].apply(lambda x: self._get_aliases_for_one_entity(x))

    def save_df(self, folder='./fixed_data/', comment=None):
        """
        Saves a DataFrame with the alternative translations to a TSV file.

        :param folder : str, The directory path where the DataFrame should be saved. Defaults to './fixed_data/'.
        :param comment : str, suffix to include in the filename:
                 - if only Google Translation is used, use "nogpt"
                 - if both Google Translation and ChatGPT are used, use "20prompt" (or the respectve number of prompts)
        """
        fname = f'{self.lang}_{self.relation}{"_" + comment if comment is not None else "_default"}.tsv'
        self.df.fillna('ERROR').to_csv(folder+fname, sep='\t', index=False)


    def add_prompt_obj_split(self, cutoff=None):
        """
        Splits the full sentence into prompt ("London is the capital of...") and object ("...the UK") parts.

        :param cutoff: str, a recurring string that can be ignored while detecting the object.

        :return: 'gt_prompt', 'gt_infl_object', ('chatgpt_prompt', 'chatgpt_infl_object' if ChatGPT) columns in the df
        """
        self.df['gt_prompt'], self.df['gt_infl_object'] = zip(*self.df.apply(lambda row: self._mask_object(row['gt_translation'], row[f'obj_label'], cutoff=cutoff), axis=1)) #{self.lang}_
        if 'chatgpt_translation' in self.df.columns:
            self.df['chatgpt_prompt'], self.df['chatgpt_infl_object'] = zip(*self.df.apply(lambda row: self._mask_object(row['chatgpt_translation'], row[f'obj_label'], cutoff=cutoff), axis=1)) #{self.lang}_

    def get_sample_for_chatgpt_fewshot(self, n=20, tgt_lang='Russian', use_translation=True):
        """
        Generates a prompt for translating templated English sentences in a controlled manner.
        1. samples the translations from the Google Translation outputs
        2. creates n-shot examples for ChatGPT (each shot consisting of source sentence, subject and object translations, and expected translation), + system prompt.
        3. This is an auxiliary function that saves time for the manual annotation of the few-shot examples.
            YOU HAVE TO CHECK THE GOOGLE TRANSLATION OUTPUTS, specifically for their adherence to subject and object translations!
            SAVE THE RESULTS IN THE `chatgpt_prompts/fewshot{n}` FOLDER!

        :param n (int): The maximum number of examples to include in the prompt. Defaults to 20.
        :param tgt_lang (str): The target language for translation. Defaults to 'Russian'.
        :param use_translation (bool, optional): Whether to include Google translations in the prompt. Defaults to True.

        Returns:
            str: A formatted string containing the translation task description along a system prompt for ChatGPT.
        """
        sample_size = min(n, len(self.df)//4)
        sample_df = self.df.sample(n=sample_size)
        sample_entities = sample_df[['en_template', f'sub_label', f'obj_label']].values.tolist() #{self.lang}_

        if use_translation:
            en_tgt_dict = self.df.set_index('en_template')['gt_translation'].to_dict()

        full_prompt = ''
        head = f"""
        You are a professional English-{tgt_lang} translator. 
        You are given the English sentences about subjects and objects. You are also given translations of subjects and objects separately. 
        You need to translate full sentences to {tgt_lang}. When translating, you have to use the translated subjects and objects. 
        Pay special attention to grammatical agreement between the words in the translated sentences.
        When translating, follow the examples:
        """
        full_prompt += head
        for entity in sample_entities:
            src_sent, sub_label, obj_label = entity
            if use_translation:
                tgt_sent = en_tgt_dict[src_sent]
            else:
                tgt_sent = ''
            prompt_element = f"""
        Source sentence: {src_sent}
        Subject translation: {sub_label}
        Object translation: {obj_label}
        Translation: {tgt_sent}
        """
            full_prompt += prompt_element
        return full_prompt #sample_entities

    def save(self, path):
        '''
        auxiliary saving function
        '''
        self.df.to_csv(path, sep='\t', index=False)
        return None

    def _unpack_data(self, trial=False):
        """
        Unpacks and processes MLAMA data for a particular language and relation.

        The method loads data templates corresponding to specified languages and relations,
        fills these templates with various sets of placeholders, and generates comparative data points.
        It extracts and processes key multilingual information, applying special handling rules
        based on language characteristics (e.g., conversion to simplified Chinese for 'zh').
        Main data are saved to a DataFrame. As a current workaround, the method also saves a formatted Excel file
        for Google Translate translation.

        :param trial (bool): If True, limits the size of the extracted data to the first 10 rows.

        :return: df_extracted: DataFrame, Processed and formatted MLAMA data points.
        :return: template_dict: dict, A dictionary with language keys pointing to their respective template subsets.
        """
        ml = MLama(self.init_path)
        ml.load()
        ml.fill_all_templates("x")

        lang_subset = ml.data[self.lang][self.relation]
        en_subset = ml.data['en'][self.relation]

        template_dict = {"en": en_subset['template'], self.lang: lang_subset['template']}
        en_triples, lang_triples = en_subset['triples'], lang_subset['triples']

        ml.fill_all_templates("x")
        lang_filled_aux = ml.data[self.lang][self.relation]['filled_templates']
        ml.fill_all_templates("xy")
        en_filled = ml.data['en'][self.relation]['filled_templates']
        lang_filled = ml.data[self.lang][self.relation]['filled_templates']

        subset_keys = list(set(en_triples.keys()).intersection(set(lang_triples.keys())))

        data_points = []
        for key in subset_keys:
            data_point = [en_triples[key]['sub_uri'], en_triples[key]['sub_label'], lang_triples[key]['sub_label'],
                          en_triples[key]['obj_uri'], en_triples[key]['obj_label'], lang_triples[key]['obj_label'],
                          en_filled[key], lang_filled[key], lang_filled_aux[key]]
            data_points.append(data_point)

        df_extracted = pd.DataFrame(data_points, columns=['sub_uri', 'en_sub_label', f'sub_label', 'obj_uri',
                                                          'en_obj_label', f'obj_label', 'en_template', f'template', f'template_aux'])

        df_extracted[f'template_prompt'] = df_extracted[f'template_aux'].apply(lambda x: x.split('[Y]')[0])#)x.rstrip('[Y].'))
        if self.lang == 'zh':
            # since Wikidata usage of Chinese is inconsistent (Traditional/Simplified), we convert everything to simplified.
            df_extracted[f'sub_label'] = df_extracted[f'sub_label'].apply(lambda x: HanziConv.toSimplified(x))
            df_extracted[f'template'] = df_extracted[f'template'].apply(lambda x: HanziConv.toSimplified(x))
            df_extracted[f'obj_label'] = df_extracted[f'obj_label'].apply(lambda x: HanziConv.toSimplified(x))
            df_extracted[f'template_prompt'] = df_extracted[f'template_prompt'].apply(lambda x: HanziConv.toSimplified(x))
        df_extracted.drop(f'template_aux', axis=1, inplace=True)
        if trial:
            df_extracted = df_extracted[:10]
        df_extracted['en_template'].to_excel(f'{self.gt_intermediate_folder}/src/{self.lang}_{self.relation}.xlsx')
        return df_extracted, template_dict

    def _translate_one_sentence(self, sentence):
        '''translates one sequence with Google Translate. Currently not working due to poor quality of non-paid Google Translate API.
        :param sentence: str
        :returns translated: str
        '''
        translated = GoogleTranslator(source='en', target=self.mlama2google[self.lang]).translate(sentence)
        print(sentence, translated, sep='\t')
        return translated

    def _chatgpt_prompt_one_entity(self, datapoint_dict, system_prompt, mode='translation'):
        """
        Generate a response from OpenAI's GPT model based on a specific prompt and mode.

        This method generates a prompt based on the provided datapoint dictionary, sends
        it to the OpenAI GPT model, and returns the content of the assistant's response.

        :param datapoint_dict (dict): A dictionary containing the input data to generate the assistant's prompt.
        :param system_prompt (str): A string representing the system-level instructions for the assistant.
        :param mode (str): Specifies the mode of assistant operation. Defaults to 'translation'.

        :return: str: The content of the response message from ChatGPT.
        """
        assistant_prompt = self._create_assistant_prompt(datapoint_dict, mode=mode)
        #print(assistant_prompt)
        completion = self.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "assistant",
                "content": assistant_prompt
            }
        ]
    )
#        print(completion)
        return completion.choices[0].message.content

    def _create_assistant_prompt(self, datapoint, mode='translation'):
        """
        Creates a formatted prompt for a single data-point.

        The method generates prompts for translation of the facts. Depending on the mode,
        it formats the prompt differently to include varying degrees of contextual or translation information.

        :param datapoint: dict A single data-point dictionary containing keys and values needed to generate the prompt.
        :param mode: str, determines the structure and content of the generated prompt. Possible values:
            - 'triplet': Includes only the subject, relation, and object information.
            - 'triplet+context': Adds detailed summaries for the subject and object.
            - 'translation' (used in the experiment): Uses a full sentence and constraints based on given translations
              of subject and object.

        :return: prompt (str), a single datapoint-based prompt.

        """
        if mode == 'triplet':
            # only giving the subject, relation and object
            prompt = f"""Subject: {datapoint['sub_label']}, relation: {datapoint['template']}, object: {datapoint['obj_label']}
             Sentence: """
        elif mode == 'triplet+context':
            # enriching the input with additional information about subjects and objects
            prompt = f"""Subject: {datapoint['sub_label']}, relation: {datapoint['template']}, object: {datapoint['obj_label']}
             Subject summary: {datapoint['sub_summary']}
             Object summary: {datapoint['obj_summary']}
             Sentence: """
        elif mode == 'translation':
            # used for the main experiment. using the full sentence and constrains (given translations) for subject and object
            prompt = f"""Source sentence: {datapoint['en_templated_sent']}
                Subject: {datapoint['sub_label']}
                Object: {datapoint['obj_label']}
             Sentence: """
        else:
            raise 'NotImplementedError'
        return prompt

    def _mask_object(self, sentence, object, threshold=.7, cutoff=None):
        """
        Masks the occurrence of an object in a sentence based on lemmas (or exact match for Chinese/Vietnamese).
        1. lemmatizes the sentence
        2. for object names like "Russian language", cutting the repetitive and non-informative parts of the name ("language")
        3. finds the object lemmas in the sentence and saves the start and end indices of the object in the sentence.
        4. if the object is found, returns part of the sentence before the object and the object itself.

        :param sentence: str, The input sentence in which the object is to be masked.
        :param object: str, The object to be masked in the sentence.
        :param threshold: float, optional A Levenshtein normalized similarity threshold value used for object detection.
            Defaults to 0.7.
        :param cutoff: str, optional, A string used to truncate the object for specific cases when objects contain parts
            that aren't uniform due to translation inconsistencies. Defaults to None.
            Apart from the word to cut itself, the ":" position specifies:
                - whether the beginning of the object should be truncated (e.g., "lingua :" for "lingua latina")
                - or the end of the object (e.g., ': язык' for 'Русский язык')

        Returns:
        :return: masked_sent, str, The part of the sentence before the object.
        :return: obj, the object itself.

        Raises:
        UnboundLocalError
            If the masking process cannot determine the indices of the object in the given sentence.
        """
        if self.lang == 'zh' or self.lang == 'vi':
            # both languages do not have much declination, therefore surface substring match (w/o lemmatization) should suffice
            masked_sent, obj = 'ERROR', 'ERROR'
            for idx in range(len(sentence))[::-1]:
                sent_suffix = sentence[idx:]
                #print(sent_suffix)
                if object in sent_suffix:
                    #print(f'found object: {object} in {sent_suffix}')
                    mask_start = idx
                    masked_sent, obj = sentence[:idx], object
                    #print(masked_sent)
                    break
            return masked_sent, obj
        else:
            # for other languages: use lemmatized sentences and objects to find the char indices of declined object
            doc = self.lemmatizer(sentence)
            masked_sent = ''
            if cutoff is not None:
                # objects for some relations consistently contain parts that are not met in translations:
                # e.g. object name: "Russian language", translation: "X's native language is Russian".
                # We only have to find "Russian" and we don't mind not having "language" there.
                # This is solved by "cutoff" parameter
                if cutoff.startswith(':'):
                    cutoff = cutoff[1:]
                    #print('cutoff: ', cutoff)
                    object = object.rstrip(cutoff)
                elif cutoff.endswith(':'):
                    cutoff = cutoff[:-1]
                    #print('cutoff: ', cutoff)
                    object = object.lstrip(cutoff)
                #print('new object: ', object)
            # lemmatizing the object
            tokenized_obj = [word.lemma for word in self.lemmatizer(object).sentences[0].words]

            masked_elements = []
            max_similarity = 0
            for word in doc.sentences[0].words:
                # iterating through the sentence to find all the object lemmas
                if word.lemma is None:
                    word.lemma = ''
                word_sim = textdistance.levenshtein.normalized_similarity(word.lemma.lower(), tokenized_obj[0].lower())
                # print(word.lemma.lower(), tokenized_obj[0].lower(), word_sim(word.lemma.lower(), tokenized_obj[0].lower()))
                max_similarity = max(max_similarity, word_sim)
                if word_sim >= threshold or word_sim >= max_similarity:
                    # if similarity between lemmas is high - save the start char id
                    mask_start = word.start_char

                # if word.lemma.lower() == tokenized_obj[-1].lower():
                if textdistance.levenshtein.normalized_similarity(word.lemma.lower(),
                                                                  tokenized_obj[-1].lower()) >= threshold:
                    # save the end char id
                    mask_end = word.end_char
                    print(f'new mask_end: {mask_end}')
            # TODO: generalize to sentences that are not ending with object
            try:
                # cut the sentence by the id of the beginning of the object
                masked_sent, obj = sentence[:mask_start], sentence[mask_start:mask_end]
                return masked_sent, obj
            except UnboundLocalError:
                return 'ERROR', 'ERROR'

    def _get_alias_pool(self):
        """
        Retrieves a mapping of object URI to their aliases from Wikidata.

        :return: idx2alias, dict, where keys are object URIs (str), and values are a string
                 of aliases concatenated with a delimiter '|||' for each entity.

        """
        # get full column from df and leave only unique ids
        id_set = set(self.df['obj_uri'].tolist())
        idx2alias = {}
        for idx in id_set:
            entity = self.wikidata_client.get(idx, load=True)
            aliases_list = []
            aliases_list.append(entity.data['labels']['en']['value'])
            try:
                aliases_dict = entity.data['aliases'][self.lang]
                aliases_list.extend([d['value'] for d in aliases_dict])
            except KeyError:
                pass
            idx2alias[idx] = '|||'.join(aliases_list)
        return idx2alias

    def _get_aliases_for_one_entity(self, wiki_id):
        '''
        retrieves the aliases for one entity from the idx2alias dictionary.
        '''
        return self.idx2alias[wiki_id]

