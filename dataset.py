from torch.utils.data import Dataset
import tqdm
import torch
import random

from tokenizer import Tokenizer
from word_transform import transform_word

class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len=50, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                self.corpus_lines = 0
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [line[:-1]
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(0, self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        sent = self.get_corpus_line(item)
        tokens, output_label, tokens_char = self.random_word(sent)

        bert_input = tokens[:self.seq_len]
        bert_label = output_label[:self.seq_len]
        bert_input_char = tokens_char[:self.seq_len]

        padding = [self.vocab.pad for _ in range(self.seq_len - len(bert_input))]
        char_padding = [self.vocab.tokenize_special_char(self.vocab.pad) \
            for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding)
        bert_label.extend(padding)
        bert_input_char.extend(char_padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "bert_input_char": bert_input_char}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        tokens_char = []

        for i, token in enumerate(tokens):
            prob_change_word = random.random()
            cur_word = token
            if prob_change_word < 0.5:
                prob = random.random()

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask
                    prob_change_char = random.random()
                    if prob_change_char < 0.6:
                        cur_word = transform_word(cur_word)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab.word_stoi))
                    cur_word = self.vocab.word_itos[tokens[i]]

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.word_stoi.get(token, self.vocab.unk)

                output_label.append(self.vocab.word_stoi.get(token, self.vocab.unk))

            else:
                tokens[i] = self.vocab.word_stoi.get(token, self.vocab.unk)
                output_label.append(0)
            tokens_char.append(self.vocab.tokenize_onehot(cur_word))

        return tokens, output_label, tokens_char

    def get_corpus_line(self, item):
        if self.on_memory:
            line = self.lines[item]
        else:
            try:
                line = self.file.__next__()
            except:
                line = None
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

        return line
