# -*- coding: utf-8 -*-
import argparse
import os
import json
from tqdm import tqdm
import csv

from dictionary import WordDictionary
from corpus import ParallelCorpus
from model import get_pred_from_api
from tokenizer import *
from prompt import *
from grammar import GrammarBook  # <-- 1. Import new GrammarBook class
from prompt import construct_prompt_za2zh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Linguistic resources
    parser.add_argument('--src_lang', type=str, default='za')
    parser.add_argument('--tgt_lang', type=str, default='zh')
    parser.add_argument('--dict_path', type=str, default='./data/dictionary_za2zh.jsonl')
    parser.add_argument('--corpus_path', type=str, default='./data/parallel_corpus.json')
    parser.add_argument('--test_data_path', type=str, default='./data/test_data.json')
    # --- 2. Add grammar_book_path parameter ---
    parser.add_argument('--grammar_book_path', type=str, default='./data/grammar_book.json',
                        help="Path to the grammar book.")

    # Config for prompt
    parser.add_argument('--prompt_type', type=str, default='za2zh', help="Should be 'za2zh' for this project.")
    parser.add_argument('--num_parallel_sent', type=int, default=5,
                        help="Number of similar sentences to include in the prompt.")

    # Output path
    parser.add_argument('--output_path', type=str, default=None, help="Path to save the detailed .jsonl log file.")
    parser.add_argument('--submission_path', type=str, default='./submission.csv',
                        help="Path to save the final submission CSV file.")

    args = parser.parse_args()

    # Load linguistic resources
    print("Loading dictionary...")
    dictionary = WordDictionary(args.src_lang, args.tgt_lang, args.dict_path)
    print("Loading parallel corpus...")
    parallel_corpus = ParallelCorpus(args.src_lang, args.tgt_lang, args.corpus_path)
    # --- 3. Load grammar book ---
    grammar_book = GrammarBook(args.grammar_book_path)
    print("Loading test data...")
    test_data = json.load(open(args.test_data_path, "r"))

    # Construct prompt function
    prompt_func = None
    prompt_type_to_prompt_func = {
        'za2zh': construct_prompt_za2zh,

    }

    if args.prompt_type not in prompt_type_to_prompt_func:
        raise NotImplementedError("Unsupported prompt type!")
    else:
        prompt_func = prompt_type_to_prompt_func[args.prompt_type]

    # Output path setup
    if args.output_path is None:
        # Add "_with_grammar" in filename for distinction
        args.output_path = "./output/api_pred_{}_parallel_{}_with_grammar.jsonl".format(args.prompt_type,
                                                                                        args.num_parallel_sent)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print("Detailed log will be saved to: {}".format(args.output_path))
    log_file = open(args.output_path, "w")

    submission_results = []

    # Run test
    for item in tqdm(test_data, desc="Translating sentences"):
        src_sentence = item[args.src_lang]

        # --- 4. Update function call, pass grammar_book ---
        prompt = prompt_func(src_sentence, dictionary, parallel_corpus, grammar_book, args)
        pred = get_pred_from_api(prompt)

        # Write to log file
        log_entry = {
            "query": src_sentence,
            "pred": pred,
            "gold": item[args.tgt_lang],
            "prompt": prompt,
            "source": item.get('source', 'unknown')
        }
        log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        submission_results.append({'id': item['id'], 'translation': pred})

    log_file.close()
    print("Detailed log saved to {}".format(args.output_path))

    # Write CSV file
    print("Writing submission file to {}...".format(args.submission_path))
    with open(args.submission_path, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=['id', 'translation'])
        writer.writeheader()
        writer.writerows(submission_results)

    print("Submission file successfully generated!")
