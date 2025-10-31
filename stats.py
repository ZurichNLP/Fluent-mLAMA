import os

import numpy as np
import pandas as pd
from collections.abc import Iterable
import re
import ast
import matplotlib.pyplot as plt
from sacrebleu import TER, CHRF
from tabulate import tabulate
from scipy import stats

from reader import MLama


class Statistics:
    def __init__(self, stats_folder='./stats/', prompt_folder='./chatgpt_prompts/translation20/'):
        self.stats_folder = stats_folder
        self.prompt_folder = prompt_folder
        self.ter, self.chrf = TER(), CHRF()

    def check_completion_status(self, langs=['ru', 'cs', 'uk', 'hr'], rels=[
                                                                            'P101', 'P103', 'P108', 'P127', 'P1376',
                                                                            'P159', 'P19', 'P20', 'P264', 'P36', 'P364',
                                                                            'P407', 'P449', 'P463', 'P495', 'P740']):
        for lang in langs:
            for rel in rels:
                if not os.path.exists(f'{self.stats_folder}{lang}_{rel}_template_gt_chatgpt.tsv'):
                    print(lang, rel)

    def count_additional_stats(self, param='adversarial', output_mode='dict', output_content='all',
                                                                            langs=['ru', 'cs', 'uk', 'hr'], rels=[
                                                                            'P101', 'P103', 'P108', 'P127', 'P1376',
                                                                            'P159', 'P19', 'P20', 'P264', 'P36', 'P364',
                                                                            'P407', 'P449', 'P463', 'P495', 'P740'],
                                                                            twomodes=False):
        # adversarial, prompts, both
        file_suffix = 'template_gt_chatgpt' if not twomodes else 'template_gt'

        stats_dict = {}
        for lang in langs:
            for rel in rels:
                if param == 'adversarial':
                    threshold = 50
                    curr_df = pd.read_csv(f'{self.stats_folder}{lang}_{rel}_{file_suffix}.tsv', sep='\t')

                    def flatten(xs):
                        for x in xs:
                            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                                yield from flatten(x)
                            else:
                                yield x

                    adv_cols = ['obj_label', 'gt_infl_object', 'chatgpt_infl_object', 'aliases'] if not twomodes else ['obj_label', 'gt_infl_object', 'aliases']
                    adv_pool = curr_df[adv_cols].values.tolist()

                    adv_pool = set([adv for adv in flatten(adv_pool)])
                    adv_pool = set([adv for adv in adv_pool if type(adv) == str and len(adv) > 0])
                    if len(adv_pool) >= threshold:
                        len_adv = threshold
                    else:
                        len_adv = len(adv_pool)
                    stats_dict[(lang, rel)] = len_adv

                elif param == 'prompts':
                    threshold = 20
                    with open(f'{self.prompt_folder}{lang}_{rel}.txt', 'r', encoding='utf-8') as f:
                        prompt_text = f.read()
                    prompt_items = re.findall(r'Source sentence: ', prompt_text)
                    stats_dict[(lang, rel)] = len(prompt_items)

                elif param == 'size':
                    curr_df = pd.read_csv(f'{self.stats_folder}{lang}_{rel}_{file_suffix}.tsv', sep='\t')
                    stats_dict[(lang, rel)] = curr_df.shape[0]

                elif param == 'num_formal_naming':
                    curr_df = pd.read_csv(f'{self.stats_folder}{lang}_{rel}_{file_suffix}.tsv', sep='\t')
                    formal_template = curr_df[curr_df['template_prompt'].str.contains(',') & ~(curr_df['gt_prompt'].str.contains(','))].shape[0]
                    #formal_gt = curr_df[curr_df['template_prompt'].str.contains(',')].shape[0]
                    stats_dict[(lang, rel)] = formal_template / curr_df.shape[0]

        if output_content == 'deviant':
            stats_dict = {k: v for k, v in stats_dict.items() if v != threshold}

        if output_mode == 'dict':
            return stats_dict
            
        elif output_mode == 'df':
            
            nested_dict = {}
            for (lang, rel), value in stats_dict.items():
                if lang not in nested_dict:
                    nested_dict[lang] = {}
                nested_dict[lang][rel] = value
            
            df = pd.DataFrame.from_dict(nested_dict, orient='columns').fillna('--')
            df = df.sort_index()
            #df = df.apply(pd.to_numeric, errors='coerce')
            return df

    def _open_df(self, lang, rel, exclude_short, twomodes=False):
        if twomodes:
            filepath = f'{self.stats_folder}{lang}_{rel}_template_gt.tsv'
        else:
            filepath = f'{self.stats_folder}{lang}_{rel}_template_gt_chatgpt.tsv'
        df = pd.read_csv(filepath, sep='\t')
        if exclude_short:
            if twomodes:
                df['gt_wrong'] = df.apply(lambda row: 1 if len(str(row['gt_prompt'])) < ((len(str(row['template_prompt']))) / 3) else 0, axis=1)#[['gt_translation', 'gt_prompt', 'gt_infl_object']]
            else:
                df['gt_wrong'] = df.apply(lambda row: 1 if len(str(row['gt_prompt'])) < ((len(str(row['chatgpt_prompt'])) + len(str(row['template_prompt']))) / 6) else 0, axis=1)#[['gt_translation', 'gt_prompt', 'gt_infl_object']]
            df = df[df['gt_wrong'] == 0]
        return df

    def _open_qe_df(self, lang, rel, exclude_short, twomodes=False):
        df = pd.read_csv(f'grammeval/{lang}_{rel}_qe.tsv', sep='\t')
        df['template_true_ranks'] = df['template_true_ranks'].apply(lambda x: ast.literal_eval(x))
        df['gt_true_ranks'] = df['gt_true_ranks'].apply(lambda x: ast.literal_eval(x))
        if 'chatgpt_true_ranks' in df.columns:
            df['chatgpt_true_ranks'] = df['chatgpt_true_ranks'].apply(lambda x: ast.literal_eval(x))

        df['template_top_rank'] = df['template_true_ranks'].apply(lambda x: min(x.values()))
        df['gt_top_rank'] = df['gt_true_ranks'].apply(lambda x: min(x.values()))
        if 'chatgpt_true_ranks' in df.columns:
            df['chatgpt_top_rank'] = df['chatgpt_true_ranks'].apply(lambda x: min(x.values()))

        if exclude_short:
            if twomodes:
                df['gt_wrong'] = df.apply(lambda row: 1 if len(str(row['gt_prompt'])) < ((len(str(row['template_prompt']))) / 3) else 0, axis=1)#[['gt_translation', 'gt_prompt', 'gt_infl_object']]
            else:
                df['gt_wrong'] = df.apply(lambda row: 1 if len(str(row['gt_prompt'])) < ((len(str(row['chatgpt_prompt'])) + len(str(row['template_prompt']))) / 6) else 0, axis=1)#[['gt_translation', 'gt_prompt', 'gt_infl_object']]
            df = df[df['gt_wrong'] == 0]
        return df

    def _get_langrel_score(self, lang, rel, topk=1, output_dtype='absolute', exclude_short=True, twomodes=False):
        df = self._open_df(lang, rel, exclude_short, twomodes=twomodes)
        #return df[df['gt_wrong'] == 1]
        topk_dict = {}
        modes = ['template', 'gt']
        if not twomodes:
            modes.append('chatgpt')
        for mode in modes:
            topk_counter = 0
            true_ranks = df[f'{mode}_true_ranks'].apply(lambda x: ast.literal_eval(x))
            rank_counter = {idx: 0 for idx in range(1, 60)}
            total_ranks = []
            for ranks in true_ranks:
                for rank in ranks:
                #    if metric == 'topk':
                    if ranks[rank] <= topk:
                        topk_counter += 1
                        break

                topk_dict[mode] = topk_counter
            if output_dtype == 'relative':
                topk_dict[mode] /= len(df)
        return topk_dict, len(df)

    def _get_langrel_percentile(self, lang, rel, percentile=50, first_only=False, exclude_short=True, twomodes=False):
        df = self._open_df(lang, rel, exclude_short, twomodes=twomodes)
        percentile_dict, mean_dict = {}, {}
        mode_list = ['template', 'gt']
        if not twomodes:
            mode_list.append('chatgpt')
        for mode in mode_list:#['template', 'gt', 'chatgpt']:
            true_ranks = df[f'{mode}_true_ranks'].apply(lambda x: ast.literal_eval(x))
            rank_counter = {idx: 0 for idx in range(1, 60)}
            total_ranks = []
            for ranks in true_ranks:
                #print(ranks)
                if first_only:
                    total_ranks.append(min(ranks.values()))
                else:
                    for rank in ranks.values():
                        #print(rank)
                        total_ranks.append(rank)
                        rank_counter[rank] += 1
            percentile_dict[mode] = np.percentile(total_ranks, percentile)
            mean_dict[mode] = np.mean(total_ranks)
        return percentile_dict, mean_dict

    def _get_langrel_qe(self, lang, rel, exclude_short=True, twomodes=False):
        #    against2metric = {'top_rank': 'top_rank', 'top1': 'top1_correct', 'top3': 'top3_correct'}
        df = self._open_qe_df(lang, rel, exclude_short, twomodes=twomodes)
        modes = ['template', 'gt', 'chatgpt'] if 'chatgpt_score' in df.columns else ['template', 'gt']
        stats_dict = {mode: df[f'{mode}_score'].mean() for mode in modes}
        return stats_dict

    def get_qe_stats(self, output_format='dict', langs=['ru', 'cs', 'uk', 'hr'], rels=[
                                                                            'P101', 'P103', 'P108', 'P127', 'P1376',
                                                                            'P159', 'P19', 'P20', 'P36', 'P364',
                                                                            'P407', 'P449', 'P463', 'P495', 'P740'],
                                                                            exclude_short=True, twomodes=False):
        template_stats, gt_stats, chatgpt_stats = {}, {}, {}
        for lang in langs:
            template_stats[lang], gt_stats[lang], chatgpt_stats[lang] = {}, {}, {}
            for rel in rels:
                stats = self._get_langrel_qe(lang, rel, exclude_short=exclude_short, twomodes=twomodes)

                template_stats[lang][rel] = stats['template']
                gt_stats[lang][rel] = stats['gt']
                if not twomodes:
                    chatgpt_stats[lang][rel] = stats['chatgpt']
                else:
                    chatgpt_stats[lang][rel] = None
        if output_format == 'dict':
            pass
        elif output_format == 'df':
            def dict2df(d):
                df = pd.DataFrame.from_dict(d, orient='columns')
                df = df.sort_index()
                return df

            template_stats, gt_stats, chatgpt_stats = dict2df(template_stats), dict2df(gt_stats), dict2df(chatgpt_stats)

        return template_stats, gt_stats, chatgpt_stats

    def get_all_stats(self, metric='topk', topk=1, percentile=50, output_dtype='relative', output_format='dict', langs=['ru', 'cs', 'uk', 'hr'], rels=[
                                                                            'P101', 'P103', 'P108', 'P127', 'P1376',
                                                                            'P159', 'P19', 'P20', 'P264', 'P36', 'P364',
                                                                            'P407', 'P449', 'P463', 'P495', 'P740'],
                                                                            exclude_short=True, first_only=False, twomodes=False,
                                                                            part_of_string_compared='prompt'):
        template_stats, gt_stats, chatgpt_stats = {}, {}, {}
        for lang in langs:
            template_stats[lang], gt_stats[lang], chatgpt_stats[lang] = {}, {}, {}
            for rel in rels:
                if metric == 'topk':
                    stats = self._get_langrel_score(lang, rel, topk=topk, output_dtype=output_dtype, exclude_short=exclude_short, twomodes=twomodes)[0]
                elif metric == 'mean':
                    stats = self._get_langrel_percentile(lang, rel, first_only=first_only, percentile=percentile, exclude_short=exclude_short, twomodes=twomodes)[1]
                elif metric == 'percentile':
                    stats = self._get_langrel_percentile(lang, rel, first_only=first_only, percentile=percentile, exclude_short=exclude_short)[0]
                elif metric == 'logprob':
                    stats = self._get_langrel_logprob(lang, rel, exclude_short=exclude_short)[0]
                elif metric == 'logprob_inf':
                    stats = self._get_langrel_logprob(lang, rel, exclude_short=exclude_short)[1]
                elif metric == 'rank_logprob_alignment':
                    stats = self._get_langrel_rank_logprob_alignment(lang, rel, exclude_short=exclude_short)
                elif metric == 'infl_factor':
                    stats = self._get_langrel_inflection_factor(lang, rel, exclude_short=exclude_short)
                elif metric == 'ter' or metric == 'chrf':
                    stats = self._get_langrel_mt_metric(lang, rel, mt_metric=metric, exclude_short=exclude_short, part_of_string_compared=part_of_string_compared)
                else:
                    raise ValueError(f'Metric {metric} is not supported.')

                if metric not in ['ter', 'chrf']:
                    template_stats[lang][rel] = stats['template']
                gt_stats[lang][rel] = stats['gt']
                if not twomodes:
                    chatgpt_stats[lang][rel] = stats['chatgpt']
                else:
                    chatgpt_stats[lang][rel] = None
        if output_format == 'dict':
            pass
        elif output_format == 'df':
            def dict2df(d):
                df = pd.DataFrame.from_dict(d, orient='columns')
                df = df.sort_index()
                return df
            if metric not in ['ter', 'chrf']:
                template_stats, gt_stats, chatgpt_stats = dict2df(template_stats), dict2df(gt_stats), dict2df(chatgpt_stats)
            else:
                gt_stats, chatgpt_stats = dict2df(gt_stats), dict2df(chatgpt_stats)
                template_stats = None
        return template_stats, gt_stats, chatgpt_stats

    def make_main_graph(self, tofile=False, file_format='pdf'):
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)#figsize=(10, 5))
        #fig.tight_layout()
        for lang, coordinate in zip(['ru', 'cs', 'uk', 'hr'], [(0, 0), (0, 1), (1, 0), (1, 1)]):
            template_data, gt_data, chatgpt_data = [], [], []
            for k in range(1, 6):
                template, gt, chatgpt = self.get_all_stats(langs=[lang], metric='topk', topk=k, output_format='df')
                template_mean, gt_mean, chatgpt_mean = template.mean(), gt.mean(), chatgpt.mean()
                template_data.append(template_mean)
                gt_data.append(gt_mean)
                chatgpt_data.append(chatgpt_mean)
            for mode, color, label in zip([template_data, gt_data, chatgpt_data], ['blue', 'green', 'red'], ['template', 'GT','ChatGPT']):
                axs[coordinate].plot(list(range(1, 6)), mode, label=label, color=color)
                axs[coordinate].set_title(lang)
                if lang == 'hr':
                    axs[coordinate].legend()
        fig.suptitle('R@n Scores of Different Verbalizations for Each Language')
        fig.supxlabel('n Value')
        fig.supylabel('R@n Score')
        #fig.legend()
        #fig.text(0.5, 0.04, 'k value', ha='center')  # Shared x-axis label
        #fig.text(0.04, 0.5, 'top-k accuracy', va='center', rotation='vertical')  # Shared y-axis label
        if tofile:
            #print(f'./graphs/general.{file_format}')
            fig.savefig(f'./graphs/general.{file_format}', dpi=600)
        #else:
        fig.show()
        return None

    def make_percentile_graph(self):
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 12))

        #fig.tight_layout()
        for lang, coordinate in zip(['ru', 'cs', 'uk', 'hr'], [(0, 0), (0, 1), (1, 0), (1, 1)]):
            template_data, gt_data, chatgpt_data = [], [], []
            p_range = range(20, 91, 10)
            for p in p_range:
                template, gt, chatgpt = self.get_all_stats(langs=[lang], metric='percentile', percentile=p, output_format='df')
                template_mean, gt_mean, chatgpt_mean = template.mean(), gt.mean(), chatgpt.mean()
                template_data.append(template_mean)
                gt_data.append(gt_mean)
                chatgpt_data.append(chatgpt_mean)
            for mode, color, label in zip([template_data, gt_data, chatgpt_data], ['blue', 'green', 'red'], ['template', 'GT','ChatGPT']):
                axs[coordinate].set_yscale('log', base=10)
                axs[coordinate].plot(list(p_range), mode, label=label, color=color)
                axs[coordinate].set_title(lang)
                if lang == 'hr':
                    axs[coordinate].legend()
        fig.suptitle('Comparison of rank percentile distribution for different languages')
        fig.supxlabel('percentile of ranks')
        fig.supylabel('average rank within percentile')
        #fig.legend()
        #fig.text(0.5, 0.04, 'k value', ha='center')  # Shared x-axis label
        #fig.text(0.04, 0.5, 'top-k accuracy', va='center', rotation='vertical')  # Shared y-axis label
        fig.show()
        return None


    def make_rank_distribution_graph(self, tofile=True, file_format='pdf', first_only=False, rels=[
                                                                            'P101', 'P103', 'P108', 'P127', 'P1376',
                                                                            'P159', 'P19', 'P20', 'P36', 'P364',
                                                                            'P407', 'P449', 'P463', 'P495', 'P740'], poster=False):
        fig, axs = plt.subplots(2, 2, sharey=True) #, figsize=(6.5, 6.5)
        for lang, coordinate in zip(['ru', 'cs', 'uk', 'hr'], [(0, 0), (0, 1), (1, 0), (1, 1)]):
            template_stats, gt_stats, chatgpt_stats = self._get_lang_rank_distribution_data(lang, first_only=first_only, rels=rels)#[], [], []

            print(len(template_stats), len(gt_stats), len(chatgpt_stats))
            colors = ['blue', 'green', 'red']
            #for mode, color, label in zip([template_stats, gt_stats, chatgpt_stats], ['blue', 'green', 'red'], ['template', 'GT','ChatGPT']):
            #axs[coordinate].set_aspect(1.05)
            axs[coordinate].boxplot([template_stats, gt_stats, chatgpt_stats], showmeans=True, showfliers=False, patch_artist=False, tick_labels=['template', 'GT','ChatGPT'])# widths=[0.1, 0.1, 0.1], boxprops=dict(facecolor=[(0, 0, 151), (0, 151, 0), (151, 0, 0)]), medianprops=dict(color='black'))
            if not poster:
                axs[coordinate].set_title(lang)
            else:
                axs[coordinate].set_title(lang, y=1.0, pad=-14)

        if first_only:
            highest = 'Highest '
        else:
            highest = ''
        if poster:
            naming = ', Slavic Languages'
            file_suffix = '_poster'
            fig.tight_layout()
        else:
            naming = ' for Each Language'
            file_suffix = ''
        #if not poster:
            fig.suptitle(f'{highest}Rank Distribution of Different Verbalizations{naming}')
            fig.supxlabel('Verbalization Type')
            fig.supylabel('Rank')
        #fig.tight_layout()
        if tofile:
            fig.savefig(f'./graphs/rank_boxplot{file_suffix}.{file_format}', dpi=600)#, dpi=300)
        #else:
        fig.show()
        return None

    def make_mt_r1_graph(self, tofile=True, file_format='pdf', mt_metric='chrf', part_of_string_compared='prompt', rels=[
                                                                            'P101', 'P103', 'P108', 'P127', 'P1376',
                                                                            'P159', 'P19', 'P20', 'P36', 'P364',
                                                                            'P407', 'P449', 'P463', 'P495', 'P740']):
        _, templateVSgt, templateVSchatgpt = self.get_all_stats(output_format='df', metric=mt_metric, rels=rels, part_of_string_compared=part_of_string_compared)
        template, gt, chatgpt = self.get_all_stats(output_format='df', metric='topk', topk=1, rels=rels)

        for lang, color in zip(['ru', 'cs', 'uk', 'hr'], ['red', 'blue', 'green', 'yellow']):
            pearsonr = round(stats.pearsonr(templateVSgt[lang], (gt - template)[lang]).statistic, 3)
            label = f'{lang}, R={pearsonr}'
            plt.scatter(templateVSgt[lang], (gt - template)[lang], c=color, marker='s', label=label)
            plt.scatter(templateVSchatgpt[lang], (chatgpt - template)[lang], c=color, marker='^')
            #text = f'{stats.pearsonr(templateVSgt[lang], (gt - template)[lang]).statistic}'
            #print(text)

        metric2label = {'chrf': 'chrF', 'bleu': 'BLEU', 'ter': 'TER'}
        plt.title(f'{metric2label[mt_metric]} Score VS R@1 Delta, By Language and Relation')
        xlabel = f'{metric2label[mt_metric]} Score, Full Sentence' if part_of_string_compared == 'full' else f'{metric2label[mt_metric]} Score, Prompt'
        plt.xlabel(xlabel)
        plt.ylabel('R@1 Delta')
        plt.legend()
        if tofile:
            plt.savefig(f'./graphs/mt_r1.{file_format}', dpi=600)
        plt.show()


    def make_logprob_rank_graph(self, tofile=True, file_format='pdf'):

        def plot_logprob_rank_distributions(data_points, show=True):
            def split_x_y(lst):
                x = np.array([el[0] for el in lst])
                y = np.array([el[1] for el in lst])
                mean_x = np.mean(x)
                mean_y = np.mean(y)
                return x, y, mean_x, mean_y

            centroid_dict = {}
            # color_map = ['red', 'blue', 'green']
            all_x = []
            for mode, color, i in zip(['template', 'gt', 'chatgpt'], ['red', 'blue', 'green'], [1, 2, 3]):
                x, y, centroid_x, centroid_y = split_x_y(data_points[mode])
                # plt.scatter(x, y, color=color, s=[2 for i in range(len(x))], label=mode)
                # if not show:
                centroid_dict[mode] = [centroid_x, centroid_y]
                if show:
                    # plt.scatter(x, y, color=color, s=[2 for i in range(len(x))], label=mode)
                    # plt.scatter(centroid_x, centroid_y, color=color, marker='s', s=5)
                    # plt.boxplot(x, i)
                    all_x.append(x)
                # m, b = np.polyfit(x, y, 1)

                # add linear regression line to scatterplot
                # plt.plot(x, m*x+b, color=color)
            if not show:
                return centroid_dict
            else:
                plt.boxplot(all_x)
                print(centroid_dict)
                template2gt_x, template2gt_y = [centroid_dict['template'][0], centroid_dict['gt'][0]], [
                    centroid_dict['template'][1], centroid_dict['gt'][1]]
                # plt.plot(template2gt_x, template2gt_y, color='black')
                template2chatgpt_x, template2chatgpt_y = [centroid_dict['template'][0], centroid_dict['chatgpt'][0]], [
                    centroid_dict['template'][1], centroid_dict['chatgpt'][1]]
                # plt.plot(template2chatgpt_x, template2chatgpt_y, color='black')
                plt.legend()
                plt.show()

        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(5.5, 5.5))

        #fig.tight_layout()
        for lang, coordinate in zip(['ru', 'cs', 'uk', 'hr'], [(0, 0), (0, 1), (1, 0), (1, 1)]):
            points = {'template': [], 'gt': [], 'chatgpt': []}
            for mode in ['template', 'gt', 'chatgpt']:
                for rel in ['P101', 'P103', 'P108', 'P127', 'P1376', 'P159', 'P19', 'P20', 'P36', 'P364', 'P407',
                            'P449', 'P463', 'P495', 'P740']:
                    data_points = self._get_langrel_rank_logprob_alignment(lang, rel, exclude_short=True)
                    curr_d = plot_logprob_rank_distributions(data_points, show=False)
                    points['template'].append(curr_d['template'])
                    points['gt'].append(curr_d['gt'])
                    points['chatgpt'].append(curr_d['chatgpt'])

                # plt.plot([curr_d['template'][0], curr_d['gt'][0]], [curr_d['template'][1], curr_d['gt'][1]], color='black', linewidth=1)
                # plt.plot([curr_d['template'][0], curr_d['chatgpt'][0]], [curr_d['template'][1], curr_d['chatgpt'][1]], color='black', linewidth=1)
            for mode, color in zip(['template', 'gt', 'chatgpt'], ['blue', 'green', 'red']):
                x, y = np.array([el[0] for el in points[mode]]), np.array([el[1] for el in points[mode]])
                mean_x, mean_y = np.mean(x), np.mean(y)

                #label = f'{lang}, {mode}'
                axs[coordinate].scatter(mean_x, mean_y, color=color, label=mode)
                axs[coordinate].set_title(lang)
                if lang == 'hr':
                    axs[coordinate].legend()
               # plt.legend()
        fig.suptitle('Avg LogP Prompt VS Avg Rank Over Languages and Modes')
        fig.supxlabel('Avg LogP Prompt')
        fig.supylabel('avg Rank')
        #fig.legend()
        fig.show()
        if tofile:
            plt.savefig(f'./graphs/logprob_rank.{file_format}', dpi=600)
#        for lang, shape in zip(['ru', 'uk', 'cs', 'hr'], ['o', 's', '*', 'X']):
#            # template_points, gt_points, chatgpt_points = [], [], []
#            points = {'template': [], 'gt': [], 'chatgpt': []}
#            for mode in ['template', 'gt', 'chatgpt']:
#                for rel in ['P101', 'P103', 'P108', 'P127', 'P1376', 'P159', 'P19', 'P20', 'P36', 'P364', 'P407', 'P449',
#                            'P463', 'P495', 'P740']:
#                    data_points = self._get_langrel_rank_logprob_alignment(lang, rel, exclude_short=True)
#                    curr_d = plot_logprob_rank_distributions(data_points, show=False)
#                    points['template'].append(curr_d['template'])
#                    points['gt'].append(curr_d['gt'])
#                    points['chatgpt'].append(curr_d['chatgpt'])
#
#
#                # plt.plot([curr_d['template'][0], curr_d['gt'][0]], [curr_d['template'][1], curr_d['gt'][1]], color='black', linewidth=1)
#                # plt.plot([curr_d['template'][0], curr_d['chatgpt'][0]], [curr_d['template'][1], curr_d['chatgpt'][1]], color='black', linewidth=1)
#            for mode, color in zip(['template', 'gt', 'chatgpt'], ['blue', 'green', 'red']):
#                x, y = np.array([el[0] for el in points[mode]]), np.array([el[1] for el in points[mode]])
#                mean_x, mean_y = np.mean(x), np.mean(y)
#
#                label = f'{lang}, {mode}'
#                plt.scatter(mean_x, mean_y, color=color, marker=shape, label=label)
#                plt.legend()
#        #    x, y = np.array([el[0] for el in gt_points]), np.array([el[1] for el in gt_points])
#        #    mean_x, mean_y = np.mean(x), np.mean(y)
#        #    plt.scatter(mean_x, mean_y, color='blue', marker=shape, label='gt')
#        #    x, y = np.array([el[0] for el in chatgpt_points]), np.array([el[1] for el in chatgpt_points])
#        #    mean_x, mean_y = np.mean(x), np.mean(y)
#        #    plt.scatter(mean_x, mean_y, color='green', marker=shape, label='chatgpt')
#            #if lang == 'hr':
#            #    plt.legend()
#        #plt.legend()
#        plt.title('Avg LogP Prompt VS Avg Rank Over Languages and Modes')
#        plt.xlabel('Avg LogP Prompt')
#        plt.ylabel('avg Rank')
#        plt.show()

    def make_general_table(self, tofile=True, wide=False, langs=['ru', 'cs', 'uk', 'hr']):
        #columns = [['template', 'gt', 'chatgpt']]
        rel_list = ['P101', 'P103', 'P108', 'P127', 'P1376', 'P159', 'P19', 'P20', 'P36', 'P364', 'P407', 'P449', 'P463', 'P495', 'P740']
        template_top1, gt_top1, chatgpt_top1 = self.get_all_stats(metric='topk', rels=rel_list, topk=1, output_format='df')
        template_top1, gt_top1, chatgpt_top1 = template_top1.mean().to_dict(), gt_top1.mean().to_dict(), chatgpt_top1.mean().to_dict()
        template_mean, gt_mean, chatgpt_mean = self.get_all_stats(metric='mean', rels=rel_list, first_only=True, output_format='df')
        template_mean, gt_mean, chatgpt_mean = template_mean.mean().to_dict(), gt_mean.mean().to_dict(), chatgpt_mean.mean().to_dict()
        lines = []
        for mode_name, mode_top1, mode_mean in zip(['template', 'gt', 'chatgpt'], [template_top1, gt_top1, chatgpt_top1], [template_mean, gt_mean, chatgpt_mean]):
            line = [mode_name]
            for lang in langs:
                line.append(round(mode_top1[lang], 3))
            for lang in langs:
                line.append(round(mode_mean[lang], 2))
            lines.append(line)
        headers = ['verbalization'] + langs + langs

        table_body = tabulate(lines, headers=headers, tablefmt='latex')
        table_body = re.sub(r'\{lrrrrrrrr\}', '{|l|rrrr|rrrr|}', table_body)
        table_parts = table_body.split('{|l|rrrr|rrrr|}')
        table_body = table_parts[0] + '{|l|rrrr|rrrr|} \n \hline & \multicolumn{4}{c|}{R@1 $\\uparrow$} & \multicolumn{4}{c|}{Mean Rank $\downarrow$} \\\\' + table_parts[1]
#        table_body = re.sub(r'\{l\|rrrr\|rrrr\|\}\n\\hline', , table_body)
        if tofile:
            fname = './tables/general_table_wide.tex' if wide else './tables/general_table.tex'
            with open(fname, 'w') as f:
                f.write(table_body)
        return table_body#tabulate(lines, headers=headers, tablefmt='latex')
#        langs = ['ru', 'cs', 'uk', 'hr']
#        for lang in langs:
#            template_stats, gt_stats, chatgpt_stats = self.get_all_stats(langs=[lang], metric='topk', topk=1, output_format='df')
#           columns.append([np.mean(template_stats), np.mean(gt_stats), np.mean(chatgpt_stats)])

    def make_general_table_additional_langs(self, tofile=True, langs=['es', 'zh', 'vi', 'id', 'da']): # 'he', 'et'
        # columns = [['template', 'gt', 'chatgpt']]
        rel_list = ['P103', 'P108', 'P159', 'P19', 'P36', 'P364', 'P407', 'P740']
        template_top1, gt_top1, _ = self.get_all_stats(metric='topk', langs=langs, rels=rel_list, topk=1,
                                                                  output_format='df', twomodes=True)
        template_top1, gt_top1 = template_top1.mean().to_dict(), gt_top1.mean().to_dict()
        template_mean, gt_mean, _ = self.get_all_stats(metric='mean', langs=langs, rels=rel_list, first_only=True,
                                                                  output_format='df', twomodes=True)
        template_mean, gt_mean = template_mean.mean().to_dict(), gt_mean.mean().to_dict()
        lines = []
        for mode_name, mode_top1, mode_mean in zip(['template', 'gt'],
                                                   [template_top1, gt_top1],
                                                   [template_mean, gt_mean]):
            line = [mode_name]
            for lang in langs:
                line.append(round(mode_top1[lang], 3))
            for lang in langs:
                line.append(round(mode_mean[lang], 2))
            lines.append(line)
        headers = ['verbalization'] + langs + langs

        table_body = tabulate(lines, headers=headers, tablefmt='latex')
        print(table_body)
        table_body = re.sub(r'\{lrrrrrrrrrr\}', '{|l|rrrrr|rrrrr|}', table_body)
        table_parts = table_body.split('{|l|rrrrr|rrrrr|}')
        table_body = table_parts[
                         0] + '{|l|rrrrr|rrrrr|} \n \hline & \multicolumn{5}{c|}{R@1 $\\uparrow$} & \multicolumn{5}{c|}{Mean Rank $\downarrow$} \\\\' + \
                     table_parts[1]
        #        table_body = re.sub(r'\{l\|rrrr\|rrrr\|\}\n\\hline', , table_body)
        if tofile:
            fname = './tables/general_table_additional_langs.tex'# if wide else './tables/general_table.tex'
            with open(fname, 'w') as f:
                f.write(table_body)
        return table_body

    def make_rank_distribution_table(self, tofile=True, first_only=False, rels=[
                                                                            'P101', 'P103', 'P108', 'P127', 'P1376',
                                                                            'P159', 'P19', 'P20', 'P36', 'P364',
                                                                            'P407', 'P449', 'P463', 'P495', 'P740']):
        columns = [['template', 'gt', 'chatgpt']]
        langs = ['ru', 'cs', 'uk', 'hr']
        for lang in langs:
            template_stats, gt_stats, chatgpt_stats = self._get_lang_rank_distribution_data(lang, first_only=first_only, rels=rels)#[], [], []
            print(len(template_stats), len(gt_stats), len(chatgpt_stats))
            columns.append([np.mean(template_stats), np.mean(gt_stats), np.mean(chatgpt_stats)])

        #col_names = ['verbalization'] + [f langs]
        df = pd.DataFrame(columns)
        df = df.T.rename(columns={0: 'verbalization', 1: 'ru', 2: 'cs', 3: 'uk', 4: 'hr'}).set_index('verbalization')
        #, orient='index', columns=['verbalization'] + langs)
        return df
         #   df = df.sort_values(by=['template', 'gt', 'chatgpt'])
         #   df = df.reset_index(drop=True)
         #   df.to_csv(f'./tables/rank_distribution_{lang}.csv')

    def make_infl_delta_table(self, tofile=True, wide=False):
        total_dict = {}

        wide_rels = ['P101', 'P103', 'P108', 'P127', 'P1376', 'P159', 'P19', 'P20', 'P36', 'P364', 'P407', 'P449',
                     'P463', 'P495', 'P740']
        narrow_rels = ['P101', 'P1376', 'P159', 'P19', 'P20', 'P407', 'P495', 'P740']

        for mode, rel_list in zip(['wide', 'narrow'], [wide_rels, narrow_rels]):
            template, gt, chatgpt = self.get_all_stats(output_format='df', metric='infl_factor', rels=rel_list)
            template_dict, gt_dict, chatgpt_dict = template.mean().to_dict(), gt.mean().to_dict(), chatgpt.mean().to_dict()
            total_dict[mode] = {'template': template_dict, 'gt': gt_dict, 'chatgpt': chatgpt_dict}

        def create_cell(narrow_value, wide_value):
            narrow_value, wide_value = round(narrow_value, 2), round(wide_value, 2)
            cell = str(narrow_value) + '_{' + str(wide_value) + '}'
            return cell

        header = ['verbalization', 'ru', 'cs', 'uk', 'hr']
        lines = []
        for mode in ['template', 'gt', 'chatgpt']:
            line = [mode]
            for lang in ['ru', 'cs', 'uk', 'hr']:
                if wide:
                    lang_cell = create_cell(total_dict['narrow'][mode][lang], total_dict['wide'][mode][lang])
                else:
                    lang_cell = round(total_dict['narrow'][mode][lang], 2)
                line.append(lang_cell)
            lines.append(line)

        table_body = tabulate(lines, header, tablefmt="latex_raw", floatfmt=(".2f", ".2f", ".2f", ".2f", ".2f"))
        table_body = re.sub(r'\{lrrrr\}', '{|l|rrrr|}', table_body)
        if tofile:
            with open('./tables/infl_delta_table.tex', 'w') as f:
                f.write(table_body)
        return table_body

    def make_relation_description_table(self, tofile=True, experiment=1):
        # columns: relation id, english template, number of facts for each lang, number of aliases differing from 50
        if experiment == 1:
            rel_list = [ 'P101', 'P103', 'P108', 'P127', 'P1376', 'P159', 'P19', 'P20', 'P36', 'P364', 'P407', 'P449', 'P463', 'P495', 'P740']
            langs = ['ru', 'cs', 'uk', 'hr']
            twomodes=False
        elif experiment == 2:
            rel_list = ['P103', 'P108', 'P159', 'P19', 'P36', 'P364', 'P407', 'P740']
            langs = ['es', 'zh', 'vi', 'id', 'da']
            twomodes=True
        ml = MLama("../mlama/data/mlama1.1/")
        ml.load()

        size_stats = self.count_additional_stats(param='size', output_mode='dict', langs=langs, rels=rel_list, twomodes=twomodes)
        print(size_stats)
        aliases_stats = self.count_additional_stats(param='adversarial', output_mode='dict', langs=langs, rels=rel_list, twomodes=twomodes)
        print(aliases_stats)
        lines = []
        for rel_id in rel_list:
            line = [rel_id]
            en_template = ml.data['en'][rel_id]['template']
            line.append(en_template)
            for lang in langs:
                line.append(size_stats[(lang, rel_id)])
            langs_aliases = {v: k[0] for k, v in aliases_stats.items() if k[1] == rel_id}
            print(langs_aliases)
            alias_line = ''
            for key in langs_aliases.keys():
                if key < 50:
                    alias_line += f'{langs_aliases[key]}: {int(key)}, '
                    #line.append(langs_aliases[key])
            if alias_line == '':
                alias_line = '50'
            alias_line = alias_line.rstrip(', ')
            line.append(alias_line)
            print(line)
            lines.append(line)

        header = ['Relation ID', 'en Template'] + langs + ['Distractor Pool']
        table_body = tabulate(lines, header, tablefmt="latex_raw")#, floatfmt=(".2f", ".2f", ".2f", ".2f", ".2f"))
        print(table_body)
        if tofile:
            with open('./tables/relation_table.tex', 'w') as f:
                f.write(table_body)
        return table_body


    def make_relation_description_table_2(self, tofile=True):
        # columns: relation id, english template, number of facts for each lang, number of aliases differing from 50
#        if experiment == 1:
        rel_list_1 = [ 'P101', 'P103', 'P108', 'P127', 'P1376', 'P159', 'P19', 'P20', 'P36', 'P364', 'P407', 'P449', 'P463', 'P495', 'P740']
        langs_1 = ['ru', 'cs', 'uk', 'hr']
#        elif experiment == 2:
        rel_list_2 = ['P103', 'P108', 'P159', 'P19', 'P36', 'P364', 'P407', 'P740']
        langs_2 = ['es', 'zh', 'vi', 'id', 'da']
        ml = MLama("../mlama/data/mlama1.1/")
        ml.load()

        size_stats_1 = self.count_additional_stats(param='size', output_mode='dict', langs=langs_1, rels=rel_list_1, twomodes=False)
        aliases_stats_1 = self.count_additional_stats(param='adversarial', output_mode='dict', langs=langs_1, rels=rel_list_1, twomodes=False)

        size_stats_2 = self.count_additional_stats(param='size', output_mode='dict', langs=langs_2, rels=rel_list_2, twomodes=True)
        aliases_stats_2 = self.count_additional_stats(param='adversarial', output_mode='dict', langs=langs_2, rels=rel_list_2, twomodes=True)

        langs = langs_1 + langs_2
        size_stats = size_stats_1 | size_stats_2
        aliases_stats = aliases_stats_1 | aliases_stats_2
        lines = []
        for rel_id in rel_list_1:
            line = [rel_id]
            en_template = ml.data['en'][rel_id]['template']
            line.append(en_template)
            langs_aliases = {k[0]: v for k, v in aliases_stats.items() if k[1] == rel_id}
            print(langs_aliases)
            small_aliases = {}
            for key, value in langs_aliases.items():
                if value < 50:
                    small_aliases[key] = value
            for lang in langs:
                #try:
                if (lang, rel_id) in size_stats.keys():
                    s = size_stats[(lang, rel_id)]
                    if lang in small_aliases.keys():
                        item = str(s) + '_{' + str(small_aliases[lang]) + '}'
                    else:
                        item = s
                else:
                    item = '--'
                line.append(item)
                #except:
                #line.append()

                    #line.append(langs_aliases[key])
            #if alias_line == '':
            #alias_line = '50'
            #alias_line = alias_line.rstrip(', ')
            #line.append(alias_line)
            print(line)
            lines.append(line)

        header = ['\makecell{Relation\\\\ID}', 'en Template'] + langs #+ ['Distractor Pool']
        table_body = tabulate(lines, header, tablefmt="latex_raw")#, floatfmt=(".2f", ".2f", ".2f", ".2f", ".2f"))
        table_body = re.sub(r'\{lllllllllll\}', '{|l|l|l|l|l|l|l|l|l|l|l|}', table_body)
        print(table_body)
        if tofile:
            with open('./tables/relation_table_all.tex', 'w') as f:
                f.write(table_body)
        return table_body


    def _get_lang_rank_distribution_data(self, lang, first_only=False, rels=[
                                                                            'P101', 'P103', 'P108', 'P127', 'P1376',
                                                                            'P159', 'P19', 'P20', 'P36', 'P364',
                                                                            'P407', 'P449', 'P463', 'P495', 'P740']):
        template_stats, gt_stats, chatgpt_stats = [], [], []
        for rel in rels:
            template_langrel, gt_langrel, chatgpt_langrel = self._get_langrel_rank_list(lang, rel, first_only=first_only, exclude_short=True)
            template_stats.extend(template_langrel)
            gt_stats.extend(gt_langrel)
            chatgpt_stats.extend(chatgpt_langrel)
        return template_stats, gt_stats, chatgpt_stats

    def _get_langrel_rank_list(self, lang, rel, first_only=False, exclude_short=True):
        df = self._open_df(lang, rel, exclude_short)
        rank_dict = {}
        for mode in ['template', 'gt', 'chatgpt']:
            true_ranks = df[f'{mode}_true_ranks'].apply(lambda x: ast.literal_eval(x))
            total_ranks = []
            for ranks in true_ranks:
                # print(ranks)
                if first_only:
                    total_ranks.append(min(ranks.values()))
                else:
                    total_ranks.extend(list(ranks.values()))
            rank_dict[mode] = total_ranks
        return rank_dict['template'], rank_dict['gt'], rank_dict['chatgpt']

    def _get_langrel_mt_metric(self, lang, rel, mt_metric='ter', part_of_string_compared='prompt', exclude_short=True, output_format='mean'):
        df = self._open_df(lang, rel, exclude_short)

        if part_of_string_compared == 'prompt':
            template_col, gt_col, chatgpt_col = 'template_prompt', 'gt_prompt', 'chatgpt_prompt'
        elif part_of_string_compared == 'full':
            template_col, gt_col, chatgpt_col = 'template', 'gt_translation', 'chatgpt_translation'
        if mt_metric == 'ter':
            df['gt_mt'] = df.apply(lambda row: self.ter.sentence_score(row[gt_col], [row[template_col]]).score, axis=1)
            df['chatgpt_mt'] = df.apply(lambda row: self.ter.sentence_score(row[chatgpt_col], [row[template_col]]).score,
                                         axis=1)
        elif mt_metric == 'chrf':
            df['gt_mt'] = df.apply(lambda row: self.chrf.sentence_score(row[gt_col], [row[template_col]]).score, axis=1)
            df['chatgpt_mt'] = df.apply(lambda row: self.chrf.sentence_score(row[chatgpt_col], [row[template_col]]).score, axis=1)

        if output_format == 'mean':
            gt_mt = df['gt_mt'].mean()
            chatgpt_mt = df['chatgpt_mt'].mean()
            return {'gt': gt_mt, 'chatgpt': chatgpt_mt}

    def _get_langrel_inflection_factor(self, lang, rel, exclude_short=True, output_format='mean'):
        df = self._open_df(lang, rel, exclude_short)

        def count_rank_difference(d, ninfl_obj):
            '''
            param d: dict of two values (non_infl: rank, infl: rank)
            '''
            try:
                if len(d) == 2:
                    infl_obj = [k for k in d.keys() if k != ninfl_obj][0]
                    rank_ninfl = d[ninfl_obj]
                    rank_infl = d[infl_obj]
                    diff = rank_ninfl - rank_infl
                    return diff
                else:
                    return None
            except:
                print(d, ninfl_obj)
                return None

        for mode in ['template', 'gt', 'chatgpt']:
            df[f'{mode}_true_ranks'] = df[f'{mode}_true_ranks'].apply(lambda x: ast.literal_eval(x))
            # print(df.columns)
            rank_colname = f'{mode}_true_ranks'
            # print(rank_colname in df.columns)
            df[f'{mode}_infl_diff'] = df.apply(
                lambda row: count_rank_difference(row[f'{mode}_true_ranks'], row.obj_label), axis=1)

        df = df.dropna(subset=['template_infl_diff', 'gt_infl_diff', 'chatgpt_infl_diff'])

        if output_format == 'mean':
            inf_dict = {'template': df['template_infl_diff'].mean(), 'gt': df['gt_infl_diff'].mean(), 'chatgpt': df['chatgpt_infl_diff'].mean()}
        elif output_format == 'list':
            inf_dict = {'template': df['template_infl_diff'].tolist(), 'gt': df['gt_infl_diff'].tolist(), 'chatgpt': df['chatgpt_infl_diff'].tolist()}
        elif output_format == 'df':
            return df
        return inf_dict

    def _get_langrel_logprob(self, lang, rel, exclude_short=True):
        df = self._open_df(lang, rel, exclude_short)
        df_noninf = df[(df['template_prompt_prob'] != -np.inf) & (df['gt_prompt_prob'] != -np.inf) & (df['chatgpt_prompt_prob'] != -np.inf)]
        logprob_dict = {'template': df_noninf['template_prompt_prob'].mean(), 'gt': df_noninf['gt_prompt_prob'].mean(), 'chatgpt': df_noninf['chatgpt_prompt_prob'].mean()}
        inf_dict = {'template': df[df['template_prompt_prob'] == -np.inf].shape[0]/df.shape[0], 'gt': df[df['gt_prompt_prob'] == -np.inf].shape[0]/df.shape[0], 'chatgpt': df[df['chatgpt_prompt_prob'] == -np.inf].shape[0]/df.shape[0]}
        return logprob_dict, inf_dict

    def _get_langrel_rank_logprob_alignment(self,  lang, rel, exclude_short=True):
        df = self._open_df(lang, rel, exclude_short)
        df_noninf = df[(df['template_prompt_prob'] != -np.inf) & (df['gt_prompt_prob'] != -np.inf) & (df['chatgpt_prompt_prob'] != -np.inf)]
        data_dict = {}
        for mode in ['template', 'gt', 'chatgpt']:
            data_points = []
            df_noninf[f'{mode}_true_ranks'] = df_noninf[f'{mode}_true_ranks'].apply(lambda x: ast.literal_eval(x))
            for idx, row in df_noninf.iterrows():
                logrel = row[f'{mode}_prompt_prob']
                highest_rank = min(row[f'{mode}_true_ranks'].values())
                data_points.append((logrel, highest_rank))
            data_dict[mode] = data_points
        return data_dict
        #pass

    def _plot_linreg(self, ax, x, y, color):
        #print(x)
        #print(y)
        b, a = np.polyfit(np.array(x), np.array(y), deg=1)
        xseq = np.linspace(min(x)-0.05, max(x)+0.05, num=100)
        ax.plot(xseq, a + b * xseq, color=color, linestyle='dotted', linewidth=1, alpha=0.6)


    def _plot_single_qe_correlation(self, plot, langs, rels, metric='top1', twomodes=False, poster=False):
        if metric == 'top1':
            metric, topk = 'topk', 1
        elif metric == 'top3':
            metric, topk = 'topk', 3
        elif metric == 'rank':
            metric, topk = 'mean', 1

        t_1, g_1, c_1 = self.get_all_stats(langs=langs, metric=metric, topk=topk, output_format='df', twomodes=twomodes,
                                        rels=rels)
        t_q, g_q, c_q = self.get_qe_stats(langs=langs, rels=rels, output_format='df', twomodes=twomodes)

        colors = ['green', 'red'] if not twomodes else ['green']
        #markers = ['D', 'X', 's', 'o', '^']
        markers = ['+', 'x', '1', '2', '*']
        total_x_g, total_y_g = [], []
        if not twomodes:
            total_x_c, total_y_c = [], []
        if poster:
            alpha = 1
        else:
            alpha = 0.7
        for lang, marker in zip(langs, markers):
            plot.scatter((g_q[lang] - t_q[lang]), (g_1[lang] - t_1[lang]), color=colors[0], marker=marker, linewidths=0.8, s=15, alpha=alpha)
            total_x_g.extend((g_q[lang] - t_q[lang]).tolist())
            total_y_g.extend((g_1[lang] - t_1[lang]).tolist())
            if not twomodes:
                plot.scatter((c_q[lang] - t_q[lang]), (c_1[lang] - t_1[lang]), color=colors[1], marker=marker, linewidths=0.8, s=15, alpha=alpha)

                total_x_c.extend((c_q[lang] - t_q[lang]).tolist())
                total_y_c.extend((c_1[lang] - t_1[lang]).tolist())

        self._plot_linreg(plot, total_x_g, total_y_g, color=colors[0])
        if not twomodes:
            self._plot_linreg(plot, total_x_c, total_y_c, color=colors[1])

        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

        handles = [f("s", colors[i]) for i in range(len(colors))]
        handles += [f(markers[i], "k") for i in range(len(langs))]

        mode_labels = ['GT', 'ChatGPT'] if not twomodes else ['GT']
        labels = mode_labels + langs

        plot.legend(handles, labels)

    def plot_qe_2_experiments(self, tofile=False, file_format='pdf', poster=False):
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(6, 6))

        self._plot_single_qe_correlation(ax1, ['ru', 'cs', 'uk', 'hr'], ['P101', 'P103', 'P108', 'P127', 'P1376',
                                                                   'P159', 'P19', 'P20', 'P36', 'P364',
                                                                   'P407', 'P449', 'P463', 'P495', 'P740'], poster=poster)
        ax1.set_title('Slavic Languages')
        self._plot_single_qe_correlation(ax2, ['es', 'zh', 'vi', 'id', 'da'],
                                   ['P103', 'P108', 'P159', 'P19', 'P36', 'P364', 'P407', 'P740'], twomodes=True, poster=poster)
        ax2.set_title('Non-Slavic Languages')
        if not poster:
            fig.suptitle('Increase Correlation Between QE Score and R@1 Score,\nPer Relation Type')
            is_poster = ''
            fig.supxlabel('QE Score Increase (Compared to Template)')
            fig.supylabel('R@1 Score Increase (Compared to Template)')

        else:
            is_poster = '_poster'
            fig.tight_layout()
        if tofile:
            fig.savefig(f'./graphs/qe{is_poster}.{file_format}', dpi=1200, format=file_format)
        else:
            fig.show()

    def calculate_qe_correlation(self, metric='top1', tofile=False):
        if metric == 'top1':
            metric, topk = 'topk', 1
        elif metric == 'top3':
            metric, topk = 'topk', 3
        elif metric == 'rank':
            metric, topk = 'mean', 1

        langs_1, langs_2 = ['ru', 'cs', 'uk', 'hr'], ['es', 'zh', 'vi', 'id', 'da']
        rels_1 = ['P101', 'P103', 'P108', 'P127', 'P1376', 'P159', 'P19', 'P20', 'P36', 'P364', 'P407', 'P449', 'P463', 'P495', 'P740']
        rels_2 = ['P103', 'P108', 'P159', 'P19', 'P36', 'P364', 'P407', 'P740']

        t_1, g_1, c_1 = self.get_all_stats(langs=langs_1, rels=rels_1, metric=metric, topk=topk, output_format='df', twomodes=False)
        t_q, g_q, c_q = self.get_qe_stats(langs=langs_1, rels=rels_1, output_format='df', twomodes=False)

        corr_dict = {}
        for lang in langs_1:
            corr_gt = stats.pearsonr((g_1[lang] - t_1[lang]), (g_q[lang] - t_q[lang]))
            corr_chatgpt = stats.pearsonr((c_1[lang] - t_1[lang]), (c_q[lang] - t_q[lang]))
            corr_dict[lang] = [(g_1[lang] - t_1[lang]).mean(), (c_1[lang] - t_1[lang]).mean(), corr_gt, corr_chatgpt]

        t_1, g_1, c_1 = self.get_all_stats(langs=langs_2, rels=rels_2, metric=metric, topk=topk, output_format='df', twomodes=True)
        t_q, g_q, c_q = self.get_qe_stats(langs=langs_2, rels=rels_2, output_format='df', twomodes=True)
        for lang in langs_2:
            corr_gt = stats.pearsonr((g_1[lang] - t_1[lang]), (g_q[lang] - t_q[lang]))
            corr_dict[lang] = [(g_1[lang] - t_1[lang]).mean(), 'N/A', corr_gt, 'N/A']

        headers = ['\makecell{Met-\\\\ric}', '\makecell{Verbali-\\\\zation}'] + langs_1 + langs_2

        def score2tabular(score, threshold=0.05):
            if score == 'N/A':
                return score
            if type(score) == np.float64:
                return round(score, 3)
            statistic, pvalue = score.statistic, score.pvalue
            tabular = str(round(statistic, 3))
            if pvalue < threshold:
                tabular += '*'
            return tabular
        line_gt_increase = ['\multirow{2}{*}{$\\Delta$}', 'GT'] + [score2tabular(corr_dict[lang][0]) for lang in langs_1 + langs_2]
        line_chatgpt_increase = [' ', 'ChatGPT'] + [score2tabular(corr_dict[lang][1]) for lang in langs_1 + langs_2]
        line_gt_corr = ['\multirow{2}{*}{$r$}', 'GT'] + [score2tabular(corr_dict[lang][2]) for lang in langs_1 + langs_2]
        line_chatgpt_corr = [' ', 'ChatGPT'] + [score2tabular(corr_dict[lang][3]) for lang in langs_1 + langs_2]
        table_body = tabulate([line_gt_increase, line_chatgpt_increase, line_gt_corr, line_chatgpt_corr], headers=headers, tablefmt='latex_raw')
        table_body = re.sub(r'\{lllllrlllll\}', '{|cl|llll|lllll|}', table_body)
        #table_parts = table_body.split('{|l|rrrr|rrrrr|}')
        #table_body = table_parts[
        #                 0] + '{|l|rrrr|rrrr|} \n \hline & \multicolumn{4}{c|}{R@1 $\\uparrow$} & \multicolumn{4}{c|}{Mean Rank $\downarrow$} \\\\' + \
        #             table_parts[1]
        #        table_body = re.sub(r'\{l\|rrrr\|rrrr\|\}\n\\hline', , table_body)
        if tofile:
            fname = './tables/qe.tex'
            with open(fname, 'w') as f:
                f.write(table_body)
        return table_body  # tabulate(lines, headers=headers, tablefmt='latex')

        #return corr_dict


