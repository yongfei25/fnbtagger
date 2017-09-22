import sys
import pathlib
from os import path
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.export import export as export_lib
from fnbtagger.train_lib import create_model_fn, read_vocab_list


def main(_):
    model_dir = 'models-dev'
    data_dir = path.join(path.dirname(__file__), '../output')
    base_dir = path.join(path.dirname(__file__), '../exports/dev')
    pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)
    token_vocabs = read_vocab_list(path.join(data_dir, 'tokens.vocab'))
    label_vocabs = read_vocab_list(path.join(data_dir, 'labels.vocab'))
    sequence_length = 30

    def serving_input_receiver_fn():
        serialized_tf_example = array_ops.placeholder(
            dtype=tf.string,
            name='input_sequence')
        receiver_tensors = {'input_sequence': serialized_tf_example}
        context_features = {
            'length': tf.FixedLenFeature([1], tf.int64)
        }
        sequence_features = {
            'tokens': tf.FixedLenSequenceFeature([sequence_length], tf.int64)
        }
        context, sequence = tf.parse_single_sequence_example(
            serialized_tf_example,
            context_features=context_features,
            sequence_features=sequence_features
        )
        features = {
            'tokens': sequence['tokens'],
            'length': context['length']
        }
        return export_lib.ServingInputReceiver(features, receiver_tensors)

    params = {
        'embedding_size': 120,
        'hidden_units': 300,
        'learning_rate': 0.001,
        'dropout_keep_prob': 1.0,
        'num_layers': 1,
        'batch_size': 1
    }
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
