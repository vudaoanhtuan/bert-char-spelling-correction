class Tokenizer:
    def __init__(self, char_vocab, word_vocab, lower_text=True):
        disable = ['vectors', 'textcat', 'tagger', 'parser', 'ner']
        self.lower_text = lower_text
        self.special = ['[pad]', '[unk]', '[bos]', '[eos]', '[mask]', '[cls]']
        char_vocab = self.special + char_vocab
        word_vocab = self.special + word_vocab
        self.pad = 0
        self.unk = 1
        self.bos = 2
        self.eos = 3
        self.mask = 4
        self.cls = 5
        self.char_itos = {i:s for i,s in enumerate(char_vocab)}
        self.word_itos = {i:s for i,s in enumerate(word_vocab)}
        self.char_stoi = {s:i for i,s in enumerate(char_vocab)}
        self.word_stoi = {s:i for i,s in enumerate(word_vocab)}
        assert self.char_stoi['[pad]'] == self.word_stoi['[pad]'] == self.pad
        assert self.char_stoi['[unk]'] == self.word_stoi['[unk]'] == self.unk
        assert self.char_stoi['[bos]'] == self.word_stoi['[bos]'] == self.bos
        assert self.char_stoi['[eos]'] == self.word_stoi['[eos]'] == self.eos
        assert self.char_stoi['[mask]'] == self.word_stoi['[mask]'] == self.mask
        assert self.char_stoi['[cls]'] == self.word_stoi['[cls]'] == self.cls
        

    def _char_tokenize(self, sent):
        if self.lower_text:
            sent = sent.lower()
        return [c for c in sent]

    def _word_tokenize(self, sent):
        if self.lower_text:
            sent = sent.lower()
        return sent.split()
        return words

    def _token_to_id(self, tokens, stoi):
        idxs = []
        for t in tokens:
            idx = stoi.get(t)
            idx = idx if idx else self.unk
            idxs.append(idx)
        return idxs

    def _id_to_token(self, idxs, itos):
        tokens = []
        for i in idxs:
            t = itos.get(i)
            t = t if t else self.special[self.unk]
            tokens.append(t)
        return tokens
    
    def process_char(self, word):
        word = self._char_tokenize(word)
        word = self._token_to_id(word, self.char_stoi)
        return word

    def process_word(self, sent):
        sent = self._word_tokenize(sent)
        sent = self._token_to_id(sent, self.word_stoi)
        return sent

    def tokenize_onehot(self, word):
        if self.lower_text:
            word = word.lower()
        token = [0.0] * len(self.char_itos)
        for c in word:
            token[self.char_stoi[c]] += 1 * 0.1
        return token
    
    def tokenize_special_char(self, char):
        token = [0.0] * len(self.char_itos)
        token[char] = 1 * 0.1
        return token