import sys
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


def main(_):
    data_dir = path.join(path.dirname(__file__), '../output')
    base_dir = path.join(path.dirname(__file__), '../exports/main')
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
        'embedding_size': 68,
        'hidden_units': 64,
        'learning_rate': 0.00061,
        'dropout_keep_prob': 0.31,
        'num_layers': 1,
        'num_epochs': 60,
    }
    model_dir = path.join('models-5', get_model_path(params))
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
    main(sys.argv)
