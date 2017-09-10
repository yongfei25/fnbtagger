import sys
import pathlib
import tensorflow as tf


def extract_tokens(sentence):
    return [x.split('/')[0].lower() for x in sentence.split(' ')]


def extract_labels(sentence):
    return [x.split('/').pop().upper() for x in sentence.split(' ')]


def make_example(sequence, labels):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(sequence)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list["tokens"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    for token, label in zip(sequence, labels):
        fl_tokens.feature.add().int64_list.value.append(token)
        fl_labels.feature.add().int64_list.value.append(label)
    return ex


class TokenIndexer:
    def __init__(self, unk='unk'):
        self.last_id = 0
        self.ids = {}
        self.tokens = {}
        self.ids[unk] = self.last_id
        self.tokens[self.last_id] = unk
        self.unk = unk

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

    def get_ids(self, tokens):
        return [self.ids.get(token, self.ids[self.unk]) for token in tokens]

    def get_tokens(self, ids):
        return [self.tokens.get(tId, self.tokens[1]) for tId in ids]


class DatasetSplitter:
    def __init__(self, split_a=0.9, split_b=0.1):
        if split_a + split_b != 1:
            raise ValueError('The split must sum up to 1')
        self.split_a = split_a
        self.split_b = split_b
        self.allocation = {
            'set_a': 0,
            'set_b': 0,
            'total': 0
        }

    def allocate(self):
        total = self.allocation['total'] + 1
        target = None
        if total == 0:
            target = 'set_a'
        elif self.allocation['set_a'] / total < self.split_a:
            target = 'set_a'
        else:
            target = 'set_b'
        self.allocation[target] += 1
        self.allocation['total'] = total
        return target


def main(_):
    data_path = './data/annotations.txt'
    train_output = 'output/train.tfrecords'
    test_output = 'output/test.tfrecords'
    dev_output = 'output/dev.tfrecords'
    pathlib.Path('./output').mkdir(parents=True, exist_ok=True)
    token_indexer = TokenIndexer(unk='unk')
    label_indexer = TokenIndexer(unk='O')
    test_splitter = DatasetSplitter(split_a=0.9, split_b=0.1)
    dev_splitter = DatasetSplitter(split_a=0.9, split_b=0.1)
    with open(data_path) as file,\
            tf.python_io.TFRecordWriter(train_output) as train_writer,\
            tf.python_io.TFRecordWriter(test_output) as test_writer,\
            tf.python_io.TFRecordWriter(dev_output) as dev_writer:
        for line in file:
            split = test_splitter.allocate()
            sequences = []
            labels = []
            if split == 'set_a':
                sequences = token_indexer.index(line, extract_tokens)
                labels = label_indexer.index(line, extract_labels)
                example = make_example(sequences, labels)
                out_string = example.SerializeToString()
                train_writer.write(out_string)
                if dev_splitter.allocate() == 'set_b':
                    dev_writer.write(out_string)
            else:
                sequences = token_indexer.get_ids(extract_tokens(line))
                labels = label_indexer.get_ids(extract_labels(line))
                example = make_example(sequences, labels)
                test_writer.write(example.SerializeToString())
    print('Done. {} train, {} test, {} dev'.format(
        test_splitter.allocation['set_a'], test_splitter.allocation['set_b'],
        dev_splitter.allocation['set_b']
    ))

if __name__ == '__main__':
    main(sys.argv)
