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
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label, t1_char_token = self.random_word(t1)
        t2_random, t2_label, t2_char_token = self.random_word(t2)

        # [CLS] tag = BOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.bos] + t1_random + [self.vocab.eos]
        t2 = t2_random + [self.vocab.eos]

        t1_label = [self.vocab.pad] + t1_label + [self.vocab.pad]
        t2_label = t2_label + [self.vocab.pad]

        t1_char_token = [self.vocab.tokenize_special_char(self.vocab.bos)] \
            + t1_char_token \
            + [self.vocab.tokenize_special_char(self.vocab.eos)]
        t2_char_token = t2_char_token + [self.vocab.tokenize_special_char(self.vocab.eos)]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]
        bert_input_char = (t1_char_token + t2_char_token)[:self.seq_len]

        padding = [self.vocab.pad for _ in range(self.seq_len - len(bert_input))]
        char_padding = [self.vocab.tokenize_special_char(self.vocab.pad) \
            for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)
        bert_input_char.extend(char_padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label,
                  "bert_input_char": bert_input_char}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, tokens):
        # tokens = sentence.split()
        output_label = []
        token_char = []

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

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.word_stoi.get(token, self.vocab.unk)

                output_label.append(self.vocab.word_stoi.get(token, self.vocab.unk))

            else:
                tokens[i] = self.vocab.word_stoi.get(token, self.vocab.unk)
                output_label.append(0)
            token_char.append(self.vocab.tokenize_onehot(cur_word))

        return tokens, output_label, token_char

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

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

        words = line[:-1].split()
        wl = len(words)
        pos = random.randint(int(0.35*wl), int(0.65*wl))
        return words[:pos], words[pos:]

    def get_random_line(self):
        if self.on_memory:
            line = self.lines[random.randrange(len(self.lines))]
        else:
            try:
                line = self.file.__next__()
            except:
                line = None
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                for _ in range(random.randint(0, self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                    self.random_file.__next__()
                line = self.random_file.__next__()
        words = line[:-1].split()
        wl = len(words)
        pos = random.randint(int(0.35*wl), int(0.65*wl))
        return words[pos:]
