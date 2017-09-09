import sys


def extract_tokens(sentence):
    return [x.split('/')[0].lower() for x in sentence.split(' ')]


def extract_labels(sentence):
    return [x.split('/').pop().upper() for x in sentence.split(' ')]


class TokenIndexer:
    def __init__(self):
        self.last_id = 0
        self.ids = {}
        self.tokens = {}

    def index(self, sentence, extract_func):
        tokens = extract_func(sentence)
        indexes = []
        for token in tokens:
            if token in self.ids:
                indexes.append(self.ids[token])
            else:
                self.last_id += 1
                self.ids[token] = self.last_id
                self.tokens[self.last_id] = token
                indexes.append(self.last_id)
        return indexes


def main(_):
    data_path = './data/annotations.txt'
    with open(data_path) as file:
        for line in file:
            print(line)

if __name__ == '__main__':
    main(sys.argv)
