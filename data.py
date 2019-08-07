import os
from io import open
import torch

from PIL import Image


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, img_dir, captions_dir):
        self.dictionary = Dictionary()
        self.img_dir = img_dir
        self.train_data = self.tokenize(os.path.join(captions_dir, 'train1_flicker.txt'))
        self.val_data  = self.tokenize(os.path.join(captions_dir, 'val1_flicker.txt'))
        self.test_data = self.tokenize(os.path.join(captions_dir, 'test_flicker.txt'))
        self.word2idx = self.dictionary.word2idx
        self.idx2word = self.dictionary.idx2word
        

    def tokenize(self, captions_dir):
        """Tokenizes a text file."""
        path = captions_dir
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            data = []
            self.dictionary.add_word("#START#")
            num_captions = 0
            max_length = 0
            for line in f:
                words = line.split()
                words = words[1:]
                max_length = max(max_length, len(words))
                num_captions = num_captions + 1
                for word in words:
                    self.dictionary.add_word(word.lower())
            
        with open(path, 'r', encoding="utf8") as f:
            print(max_length)
            data = [(0,0) for i in range(num_captions)]
            c = 0
            ind = 0
            for line in f:
                words = line.split()
                if (c == 0):
                    file_name = words[0][:-2]
                    img_temp = Image.open(os.path.join(self.img_dir, file_name))
                    img = img_temp.copy()
                    img_temp.close()

                words = ["#START#"] + [word.lower() for word in words[1:]]
                caps = [self.dictionary.word2idx[word] for i, word in enumerate(words)]
                tars = caps[1:]
                caps = caps[:-1]
                data[ind] = (img,caps,tars)
                ind = ind + 1
                c = c + 1
                if (c == 5): c = 0
        return data