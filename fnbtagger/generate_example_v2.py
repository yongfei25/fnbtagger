import sys
import pathlib
import tensorflow as tf


def extract_tokens(sentence):
    return [x.split('/')[0].lower() for x in sentence.split(' ')]


def extract_labels(sentence):
    return [x.split('/').pop().upper() for x in sentence.split(' ')]


def make_example(sequence, labels):
    if len(sequence) != len(labels):
        raise ValueError(
            'sequence and labels must have equal length. {} != {}'.format(
                len(sequence), len(labels)
            )
        )

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


class DatasetSplitter:
    def __init__(self, split=0.9):
        if split <= 0 or split >= 1:
            raise ValueError('The split must beween 0 and 1')
        self.split_a = split
        self.split_b = 1 - split
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


class TokenIndexer:
    def __init__(self, max_length=51, unk='<unk>', pad='<pad>'):
        self.ids = {}
        self.tokens = {}
        self.ids[pad] = 0
        self.tokens[0] = pad
        self.ids[unk] = 1
        self.tokens[1] = unk
        self.unk = unk
        self.pad = pad
        self.last_id = 1
        self.max_length = max_length

    def index(self, sentence, extract_func):
        tokens = extract_func(sentence)
        return self.index_tokens(tokens)

    def index_tokens(self, tokens):
        indexes = []
        for token in tokens:
            if token in self.ids:
                indexes.append(self.ids[token])
            else:
                self.last_id += 1
                self.ids[token] = self.last_id
                self.tokens[self.last_id] = token
                indexes.append(self.last_id)
        indexes = self.pad_right(indexes, self.max_length, self.ids[self.pad])
        return indexes

    def get_ids(self, tokens):
        indexes = [self.ids.get(token, self.ids[self.unk]) for token in tokens]
        return self.pad_right(indexes, self.max_length, self.ids[self.pad])

    def get_tokens(self, ids):
        indexes = [self.tokens.get(tId, self.tokens[1]) for tId in ids]
        return self.pad_right(indexes, self.max_length, self.tokens[0])

    def pad_right(self, aList, length, padding):
        cList = list(aList)
        l = len(aList)
        while l < length:
            cList.append(padding)
            l += 1
        return cList


def write_vocab(vocab_iterator, fd):
    for vocab in vocab_iterator:
        fd.write('{}\n'.format(vocab))


def main(_):
    data_path = './data/annotations.txt'
    train_output = 'output/train.tfrecord'
    test_output = 'output/test.tfrecord'
    dev_output = 'output/dev.tfrecord'
    tiny_output = 'output/tiny.tfrecord'
    tokens_output = 'output/tokens.vocab'
    labels_output = 'output/labels.vocab'
    max_length = 30
    pathlib.Path('./output').mkdir(parents=True, exist_ok=True)
    token_indexer = TokenIndexer(unk='unk', max_length=max_length)
    label_indexer = TokenIndexer(unk='O', max_length=max_length)
    test_splitter = DatasetSplitter(split=0.9)
    dev_splitter = DatasetSplitter(split=0.9)
    print_every = 1000

    with open(data_path) as file,\
            open(tokens_output, 'w') as tokens_fd,\
            open(labels_output, 'w') as labels_fd,\
            tf.python_io.TFRecordWriter(train_output) as train_writer,\
            tf.python_io.TFRecordWriter(test_output) as test_writer,\
            tf.python_io.TFRecordWriter(tiny_output) as tiny_writer,\
            tf.python_io.TFRecordWriter(dev_output) as dev_writer:
        test_examples = []
        for line in file:
            line = line.rstrip('\n')
            if line == '':
                continue
            sequences = extract_tokens(line)
            labels = extract_labels(line)
            if len(sequences) > max_length:
                continue
            if test_splitter.allocate() == 'set_a':
                # index the tokens so that we can build the vocab
                sequences_idx = token_indexer.index_tokens(sequences)
                label_idx = label_indexer.index_tokens(labels)
                example = make_example(sequences_idx, label_idx)
                out_string = example.SerializeToString()
                train_writer.write(out_string)
                if dev_splitter.allocate() == 'set_b':
                    dev_writer.write(out_string)
                if test_splitter.allocation['set_a'] <= 20:
                    tiny_writer.write(out_string)
                if test_splitter.allocation['set_a'] % print_every == 0:
                    print(' '.join(sequences))
                    print(' '.join([str(x) for x in label_idx]))
                    print('-------------------------')
            else:
                # Add test examples to write after we indexed all
                # the train examples
                test_examples.append((sequences, labels))

        for sequences, labels in test_examples:
            label_idx = label_indexer.get_ids(labels)
            sequences_idx = token_indexer.get_ids(sequences)
            example = make_example(sequences_idx, label_idx)
            out_string = example.SerializeToString()
            test_writer.write(out_string)

        write_vocab(token_indexer.tokens.values(), tokens_fd)
        write_vocab(label_indexer.tokens.values(), labels_fd)
        print('Done. {} train, {} test, {} dev'.format(
            test_splitter.allocation['set_a'],
            test_splitter.allocation['set_b'],
            dev_splitter.allocation['set_b']
        ))


if __name__ == '__main__':
    main(sys.argv)
