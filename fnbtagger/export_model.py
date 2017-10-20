import argparse
import pathlib
from os import path
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.export import export as export_lib
from fnbtagger.train_lib import (
    create_model_fn,
    read_vocab_list,
    get_model_path
)


def main(lang, step):
    data_dir = path.join(path.dirname(__file__),
                         '../output/{}'.format(lang))
    base_dir = path.join(path.dirname(__file__),
                         '../exports/{}/main'.format(lang))
    model_dir = path.join(path.dirname(__file__),
                          '../models/{}'.format(lang))
    pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)
    token_vocabs = read_vocab_list(path.join(data_dir, 'tokens.vocab'))
    label_vocabs = read_vocab_list(path.join(data_dir, 'labels.vocab'))
    sequence_length = 30

    def serving_input_receiver_fn():
        receiver_tensors = {
            'tokens': array_ops.placeholder(
                dtype=tf.int64,
                shape=[None, sequence_length],
                name='tokens'
            ),
            'length': array_ops.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='length'
            )
        }
        return export_lib.ServingInputReceiver(
            receiver_tensors, receiver_tensors)

    params = {
        'embedding_size': 38,
        'hidden_units': 119,
        'learning_rate': 0.00119,
        'dropout_keep_prob': 0.29,
        'num_layers': 1,
        'num_epochs': 70,
    }
    model_dir = path.join(model_dir, step, get_model_path(params))
    estimator = Estimator(
        model_fn=create_model_fn(
            vocab_list=token_vocabs,
            class_list=label_vocabs
        ),
        model_dir=model_dir,
        params=params
    )
    estimator.export_savedmodel(
        export_dir_base=base_dir,
        serving_input_receiver_fn=serving_input_receiver_fn,
        assets_extra={
            'vocabs': path.join(data_dir, 'tokens.vocab'),
            'labels': path.join(data_dir, 'labels.vocab')
        }
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export model')
    parser.add_argument('language', choices=['en', 'zh'])
    parser.add_argument('step', help='step name')
    args = parser.parse_args()
    main(args.language, args.step)
