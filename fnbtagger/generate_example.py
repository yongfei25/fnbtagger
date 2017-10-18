import sys
from os import path
import pathlib
import argparse
import tensorflow as tf
from fnbtagger.example_lib import TokenIndexer, DatasetSplitter


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


def write_vocab(vocab_iterator, fd):
    for vocab in vocab_iterator:
        fd.write('{}\n'.format(vocab))


def main(lang, max_length=30, test_split=0.9, dev_split=0.9):
    data_path = path.join(path.dirname(__file__),
                          '../data/annotations-{}.txt'.format(lang))
    output_path = path.join(path.dirname(__file__),
                            '../output/{}'.format(lang))
    train_output = path.join(output_path, 'train.tfrecord')
    test_output = path.join(output_path, 'test.tfrecord')
    dev_output = path.join(output_path, 'dev.tfrecord')
    tiny_output = path.join(output_path, 'tiny.tfrecord')
    tokens_output = path.join(output_path, 'tokens.vocab')
    labels_output = path.join(output_path, 'labels.vocab')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    token_indexer = TokenIndexer(unk='unk', max_length=max_length)
    label_indexer = TokenIndexer(unk='O', max_length=max_length)
    test_splitter = DatasetSplitter(test_split)
    dev_splitter = DatasetSplitter(dev_split)
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
    parser = argparse.ArgumentParser(description='Generate TFrecord files.')
    parser.add_argument('language', choices=['en', 'zh'])
    args = parser.parse_args()
    main(args.language)
