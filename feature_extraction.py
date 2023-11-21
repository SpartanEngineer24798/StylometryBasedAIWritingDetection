import argparse
import csv
import json
import os
import re
import statistics

from collections import defaultdict

import nltk
import readability
import spacy
import torch
from tqdm import tqdm
from lexicalrichness import LexicalRichness
from nltk.tokenize import word_tokenize, sent_tokenize
from spacy import util
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from zipfile import ZipFile
import requests

class PerplexityCalculator:
    STRIDE = 512
    NLP_MAX_LENGTH = 200000

    def __init__(self):
        self._tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
        self._model = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self._max_length = self._model.config.n_positions
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model.to(self._device)
        self._nlp = spacy.load(spacy_model)
        self._nlp.add_pipe("sentencizer")

        infixes = self._nlp.Defaults.infixes + ['`', "'", '"']
        self._nlp.tokenizer.infix_finditer = util.compile_infix_regex(infixes).finditer
        self._count_vectorizer_dict = {}
        self._tfidf_transformer_dict = {}
        self._tfidf_vectors = {}

    def calculate_inner(self, in_text: str, precision=6) -> float:
        encodings = self._tokenizer(in_text, return_tensors='pt', max_length=self.NLP_MAX_LENGTH, truncation=True)
        seq_len = encodings.data['input_ids'].size(1)

        nlls = []
        prev_end_loc = 0
        count = 0
        for begin_loc in range(0, seq_len, self.STRIDE):
            end_loc = min(begin_loc + self._max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self._device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self._model(input_ids, labels=target_ids)
                nlls.append(outputs.loss)

            prev_end_loc = end_loc
            count += 1
            if end_loc == seq_len:
                break
        return round(torch.exp(torch.stack(nlls).mean()).item(), precision)
    
def ppl_calculator(text: str, precision=2) -> float:
    ppl_calc = PerplexityCalculator()
    return ppl_calc.calculate_inner(text, precision)

def word_count(document):
    tokens = word_tokenize(document)
    nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
    filtered = [w for w in tokens if nonPunct.match(w)]
    return len(filtered)

def word_count_sent(document):
    tokens = sent_tokenize(document)
    nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
    filtered = [w for w in tokens if nonPunct.match(w)]
    word_counts = [word_count(sent) for sent in filtered]
    if len(word_counts) == 0:
        return 0, 0
    mean = sum(word_counts) / len(word_counts)
    if len(word_counts) < 2:
        stdev = 0
    else:
        stdev = statistics.stdev(word_counts)
    return mean, stdev

def special_punc_count_sent(document):
    special_puncts = ['!', '\"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    tokens = sent_tokenize(document)
    nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
    filtered = [w for w in tokens if nonPunct.match(w)]
    punct_count = 0
    total_sentences = len(filtered)
    if total_sentences == 0:
        return 0
    for sent in filtered:
        for char in sent:
            if char in special_puncts:
                punct_count += 1
    return float(punct_count) / total_sentences

def readability_score(document):
    try:
        r = readability.getmeasures(document, lang='en')
        fk = r['readability grades']['Kincaid']
    except Exception:
        return 0
    else:
        return fk
    
def lexical_richness(document):
    sample_size = 10
    iterations = 50
    lex = LexicalRichness(document)
    ret_list = []
    words = document.split()
    try:
        if len(words) > 45:
            ret_list.append(lex.mattr(window_size=25))
        else:
            window_size = max(1, len(words) // 3)  # Adjusted window size
            if window_size > len(words):
                window_size = len(words)
            ret_list.append(lex.mattr(window_size=window_size))
    except Exception:
        ret_list.append(0)  # Return 0 if an exception is thrown during feature extraction
    ret_list.append(lex.mtld(threshold=0.72))
    return ret_list

def extract_features(document):
    results = []

    words_per_sent = word_count_sent(document)
    results.append(words_per_sent[0])
    results.append(words_per_sent[1])

    special_punc_sent_result = special_punc_count_sent(document)
    results.append(special_punc_sent_result)

    readability_results = readability_score(document)
    results.append(readability_results)

    lexical_richness_results = lexical_richness(document)
    results.extend(lexical_richness_results)

    results.append(ppl_calculator(document))

    return results

def sanitize_title(title):
    restricted_chars = r'\/:*?"<>|'
    sanitized_title = ''.join(c if c.isalnum() or c in ['_', '-'] else '_' if c not in restricted_chars else '' for c in title)
    return sanitized_title

def process_data(data, output_csv):
    total_entries = len(data)

    progress_bar = tqdm(total=total_entries, desc="Processing Entries")

    csv_exists = os.path.exists(output_csv)

    existing_titles = set()
    if csv_exists:
        with open(output_csv, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            existing_titles.update(row['Title'] for row in csv_reader)

    with open(output_csv, 'a', newline='', encoding='utf-8') as csv_file:

        fieldnames = ['Title', 'Label', 'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not csv_exists:
            csv_writer.writeheader()

        for entry in data:
            title = entry.get('title', '')
            wiki_intro = entry.get('wiki_intro', '')
            generated_intro = entry.get('generated_intro', '')

            sanitized_title = sanitize_title(title)

            if sanitized_title in existing_titles:
                progress_bar.update(1)
                continue

            wiki_features = extract_features(wiki_intro)
            csv_writer.writerow({
                'Title': sanitized_title,
                'Label': 'human',
                'Feature1': wiki_features[0],
                'Feature2': wiki_features[1],
                'Feature3': wiki_features[2],
                'Feature4': wiki_features[3],
                'Feature5': wiki_features[4],
                'Feature6': wiki_features[5],
                'Feature7': wiki_features[6]
            })

            generated_features = extract_features(generated_intro)
            csv_writer.writerow({
                'Title': sanitized_title,
                'Label': 'ai',
                'Feature1': generated_features[0],
                'Feature2': generated_features[1],
                'Feature3': generated_features[2],
                'Feature4': generated_features[3],
                'Feature5': generated_features[4],
                'Feature6': generated_features[5],
                'Feature7': generated_features[6]
            })

            existing_titles.add(sanitized_title)
            progress_bar.update(1)

    progress_bar.close()

def ensure_output_directory(output):
    output_directory = os.path.dirname(output)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    if not os.path.exists(output):
        with open(output, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Title', 'Label', 'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7']
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()

def check_file_existence(input):
    if not os.path.exists(input):
        print(f"Error: The input file '{input}' does not exist. Please provide the file or call the program again with the argument 'download-dataset' ")
        exit(1)

def initialize():
    nltk.download('punkt')
    global spacy_model
    spacy_model = "en_core_web_trf"
    global gpt2_model
    gpt2_model = "gpt2"

def download_and_extract_gpt_wiki_intro(output_path):
    url = "https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro/resolve/main/GPT-wiki-intro.csv.zip?download=true"

    zip_file_name = "GPT-wiki-intro.csv.zip"

    full_path = os.path.join(os.getcwd(), zip_file_name)

    response = requests.get(url)
    with open(full_path, "wb") as zip_file:
        zip_file.write(response.content)

    with ZipFile(full_path, "r") as zip_ref:
        zip_ref.extractall(output_path)

    os.remove(full_path)

    print("GPT-wiki-intro.csv.zip downloaded and extracted.")
    print(full_path)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process data and extract features.')
    parser.add_argument('-i', '--input', nargs='?', default='', help='Path to the input CSV file containing data')
    parser.add_argument('-o', '--output', nargs='?', default='', help='Full path and name of the output CSV file')
    parser.add_argument('--download-dataset', action='store_true', help='Download and extract the GPT-wiki-intro dataset')
    return parser.parse_args()


def main():
    args = parse_arguments()

    initialize()
    print("Initializing...")

    if args.download_dataset:
        download_and_extract_gpt_wiki_intro(os.path.dirname(args.output))
        print("Input dataset downloaded and extracted.")
    else:
        check_file_existence(args.input)

    input_file_path = args.input

    output = args.output
    ensure_output_directory(output)
    print("Input and Output directories checked.")

    input_csv = input_file_path
    if not os.path.exists(input_csv):
        print(f"Error: The input CSV file '{input_csv}' does not exist. Please provide the correct file path.")
        exit(1)

    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        data = list(csv.DictReader(csvfile))

    print("Input CSV loaded. Beginning Data Processing")

    process_data(data, output)

    print("Process finished. Exiting...")

if __name__ == "__main__":
    main()
