from processor import Processor
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # language in question
    parser.add_argument('-l', '--lang')
    # list of relations in question, comma-delimited
    parser.add_argument('-r', '--rel')
    # either compare 2 modes (template VS GT) or 3 modes (template VS GT VS ChatGPT)
    parser.add_argument('-m', '--modes', type=int, default=3,
                        help='2: template VS GT, 3: template VS GT VS ChatGPT')
    args=parser.parse_args()
    args.rel = args.rel.split(',')
    spec = '20prompt' if args.modes == 3 else 'nogpt'
    p = Processor(model="meta-llama/Llama-2-7b-chat-hf")
    for r in args.rel:
        print(f'relation {r}')
        a = time.time()
        df = p.single_processing_run(args.lang, r, dataset_spec=spec, trial_run=False)
        b = time.time()
        print(f'task fulfilled within {b-a} seconds')
