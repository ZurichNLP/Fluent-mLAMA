import pandas as pd
from comet import download_model, load_from_checkpoint
from wikidata.client import Client
import ast
from tabulate import tabulate
import re

class QECheck:
    def __init__(self, qe_model='comet'):
        if qe_model == 'comet':
            model_path = download_model("Unbabel/wmt20-comet-qe-da")
            self.model = load_from_checkpoint(model_path)
        else:
            self.model = None
        pass

    def load_data(self, lang, rel, datapath='./stats/', suffix='template_gt_chatgpt'):
        self.lang = lang
        self.rel = rel
        #self.datapath = datapath
        self.df = pd.read_csv(f'{datapath}{lang}_{rel}_{suffix}.tsv', sep='\t')
        self.df['template_true_ranks'] = self.df['template_true_ranks'].apply(lambda x: ast.literal_eval(x))
        self.df['gt_true_ranks'] = self.df['gt_true_ranks'].apply(lambda x: ast.literal_eval(x))
        if 'chatgpt_true_ranks' in self.df.columns:
            self.df['chatgpt_true_ranks'] = self.df['chatgpt_true_ranks'].apply(lambda x: ast.literal_eval(x))

        self.df['template_top_rank'] = self.df['template_true_ranks'].apply(lambda x: min(x.values()))
        self.df['gt_top_rank'] = self.df['gt_true_ranks'].apply(lambda x: min(x.values()))
        if 'chatgpt_true_ranks' in self.df.columns:
            self.df['chatgpt_top_rank'] = self.df['chatgpt_true_ranks'].apply(lambda x: min(x.values()))

        self.wikidata_client = Client()
        return self.df

    def evaluate_langrel(self, save=False, savepath='./grammeval/'):
        #df = df.head(48)
        enVStemplate = self._zip_2_columns(self.df, 'en_template', 'template')
        enVSgt = self._zip_2_columns(self.df, 'en_template', 'gt_translation')
        enVSchatgpt = self._zip_2_columns(self.df, 'en_template', 'chatgpt_translation')

        enVStemplate_output = self.model.predict(enVStemplate, batch_size=40, gpus=1)
        enVSgt_output = self.model.predict(enVSgt, batch_size=40, gpus=1)
        enVSchatgpt_output = self.model.predict(enVSchatgpt, batch_size=40, gpus=1)
        self.df['template_score'] = enVStemplate_output.scores
        self.df['gt_score'] = enVSgt_output.scores
        self.df['chatgpt_score'] = enVSchatgpt_output.scores

        self.total_dict = {'template': enVStemplate_output.system_score, 'gt': enVSgt_output.system_score, 'chatgpt': enVSchatgpt_output.system_score}

        if save:
            self.df.to_csv(f'{savepath}{self.lang}_{self.rel}_qe.tsv', sep='\t', index=False)
        return self.df, self.total_dict

    def _zip_2_columns(self, df, col1, col2):
        list1, list2 = df[col1].to_list(), df[col2].to_list()
        return [{'src': c1, 'mt': c2} for c1, c2 in zip(list1, list2)]

    def add_gender_info(self, save=False):
        self.df['gender'] = self.df['sub_uri'].apply(lambda x: self._get_gender(x))
        if save:
            self.df.to_csv(f'grammeval_gender/{self.lang}_{self.rel}_gender.tsv', sep='\t', index=False)
        return self.df

    def add_gender_info_from_file(self, filepath='gender.tsv', save=False):
        df_gender = pd.read_csv(filepath, sep='\t')
        df_dict = dict(zip(df_gender['p'], df_gender['genderLabel']))
        self.df['gender'] = self.df['sub_uri'].apply(lambda x: df_dict[x] if x in df_dict.keys() else 'NA')
        if save:
            self.df.to_csv(f'grammeval_gender/{self.lang}_{self.rel}_gender.tsv', sep='\t', index=False)

        return self.df

    def _get_gender(self, wikidata_id):
        id2gender = {"Q6581097": "male", "Q6581072": "female", "Q1097630": "intersex", "Q1052281": "transgender female",
                     "Q2449503": "transgender male"}
        entity = self.wikidata_client.get(wikidata_id, load=True)

        try:
            gender_id = entity.data['claims']['P21'][0]['mainsnak']['datavalue']['value']['id']
            gender = id2gender[gender_id]
        except:
            gender = 'NA'

        return gender

    def rule_based_gender_check(self):
        for mode, col_name in zip(['template', 'gt', 'chatgpt'], ['template', 'gt_translation', 'chatgpt_translation']):
            self.df[f'{mode}_gender_check'] = self.df.apply(lambda row: self._rule_based_gender_check_sentence(row[col_name], row['gender']), axis=1)
        return self.df

    def _rule_based_gender_check_sentence(self, sentence, gender):
        rel2lang2gender2form = {
        'P19': {
                 'ru': {'male': ' родился ',  'transgender male': ' родился ', 'female': ' родилась ', 'transgender female': ' родилась '},
                 'cs': {'male': ' narodil ',  'transgender male': ' narodil ', 'female': ' narodila ', 'transgender female': ' narodila '},
                 'uk': {'male': ' народився ',  'transgender male': ' народився ', 'female': ' народилася ', 'transgender female': ' народилася '},
                 'hr': {'male': ' rođen ',  'transgender male': ' rođen ', 'female': ' rođena ', 'transgender female': ' rođena '}
                },
        'P20': {
                'ru': {'male': ' умер ', 'transgender male': ' умер ', 'female': ' умерла ', 'transgender female': ' умерла '},
                'cs': {'male': ' zemřel ', 'transgender male': ' zemřel ', 'female': ' zemřela ', 'transgender female': ' zemřela '},
                'uk': {'male': ' помер ', 'transgender male': ' помер ', 'female': ' померла ', 'transgender female': ' померла '},
                'hr': {'male': ' umro ', 'transgender male': ' umro ', 'female': ' umrla ', 'transgender female': ' umrla '}
            }
        }
        if gender not in rel2lang2gender2form[self.rel][self.lang].keys():
            return None
        if rel2lang2gender2form[self.rel][self.lang][gender] in sentence:
            return 1
        return 0

    def subsample(self, by='gender'):
        if by == 'gender':
            #if 'gender' not in self.df.columns:
            #    _ = self._add_gender_info()
            sample = self.df[self.df['gender'].isin(['transgender female', 'female'])].copy(deep=True)
        return sample

    def get_gender_stats_langrel(self, metric='topk', topk=1, output='stats'):
        if metric == 'topk':
            metric_name = 'top' + str(topk) + '_correct'
        elif metric == 'rank':
            metric_name = 'top_rank'

        subsample_df = self.subsample()
        if output == 'stats':
            total_stats = self.df[
                ['template_gender_check', 'gt_gender_check', 'chatgpt_gender_check', f'template_{metric_name}',
                 f'gt_{metric_name}', f'chatgpt_{metric_name}']].mean().to_dict()

            subsample_stats = subsample_df[['template_gender_check', 'gt_gender_check', 'chatgpt_gender_check', f'template_{metric_name}', f'gt_{metric_name}', f'chatgpt_{metric_name}']].mean().to_dict()
            total_dict = {'total_' + k: v for k, v in total_stats.items()} | {'female_' + k: v for k, v in subsample_stats.items()}
            return total_dict
        elif output == 'df':
            return self.df[['template_gender_check', 'gt_gender_check', 'chatgpt_gender_check', f'template_{metric_name}', f'gt_{metric_name}', f'chatgpt_{metric_name}']], subsample_df[['template_gender_check', 'gt_gender_check', 'chatgpt_gender_check', f'template_{metric_name}', f'gt_{metric_name}', f'chatgpt_{metric_name}']]

    def make_gender_stats_table(self, langs=['ru', 'cs', 'uk', 'hr'], rels=['P19', 'P20'], metric='topk', topk=1, tofile=False):
        lang_dict = {}
        for lang in langs:
            df_total, sub_total = [], []
            for rel in rels:
                self.load_data(lang, rel)
                self.add_gender_info_from_file()
                self.rule_based_gender_check()
                df, sub = self.get_gender_stats_langrel(metric=metric, topk=topk, output='df')
                print(df.shape == sub.shape)
                df_total.append(df)
                sub_total.append(sub)

            df_total = pd.concat(df_total)
            sub_total = pd.concat(sub_total)
            #print(df_total == sub_total)
            lang_dict[lang] = (df_total.mean().to_dict(), sub_total.mean().to_dict())

        headers = ['\multirow{2}{*}{Verbalization}'] + ['\multicolumn{2}{c|}{' + lang + '}' for lang in langs]
        col1 = [''] + ['\\%(F)', 'R@1(F)'] * len(langs)
        data = [headers, col1]
        metric_name = 'top' + str(topk) + '_correct' if metric == 'topk' else 'top_rank'
        for mode in ['Template', 'GT', 'ChatGPT']:
            line = [mode]
            for lang in langs:
                fem_dict = lang_dict[lang][1]
                gender = round(fem_dict[f'{mode.lower()}_gender_check'] * 100, 1)
                target = round(fem_dict[f'{mode.lower()}_{metric_name}'], 3)
                line.extend([gender, target])
            data.append(line)

        table_body = tabulate(data, tablefmt='latex_raw')
        table_body = re.sub(r'\{lllllllll\}', '{|l|ll|ll|ll|ll|}', table_body)
        table_body = re.sub('&\s+&\s+&\s+&\s+\\\\', ' \\\\', table_body )
#        table_body = re.sub('R@1(F) \\\\', r'R@1(F) \\ \hline\\', table_body)
        if tofile:
            fname = './tables/gender_stats.tex'
            with open(fname, 'w') as f:
                f.write(table_body)
        return table_body #        df = pd.DataFrame.from_dict(total_dict, orient='index')
