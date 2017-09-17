
import sys
from os import path
import pathlib
import tensorflow as tf
from tensorflow.python.estimator.estimator_lib import EstimatorSpec
from tensorflow.python.estimator.estimator_lib import Estimator
from tensorflow.contrib.learn.python.learn import Experiment

# tf.logging.set_verbosity(tf.logging.INFO)


def read_vocab_list(vocab_file_path):
    vocabs = []
    with open(vocab_file_path) as file:
        for line in file:
            vocabs.append(line.rstrip('\n'))
    return vocabs


def parseExample(record, vocab_list, embedding_size):
    # example feature
    context_features = {
        'length': tf.FixedLenFeature([], tf.int64)
    }
    sequence_features = {
        'tokens': tf.FixedLenSequenceFeature([], tf.string),
        'labels': tf.FixedLenSequenceFeature([], tf.int64)
    }
    context, sequence = tf.parse_single_sequence_example(
        record,
        context_features=context_features,
        sequence_features=sequence_features
    )
    # feature columns
    tokens_column = \
        tf.feature_column.categorical_column_with_vocabulary_list(
            key='tokens',
            vocabulary_list=vocab_list,
            dtype=tf.string
        )
    embedding_column = tf.feature_column.embedding_column(
        categorical_column=tokens_column,
        dimension=embedding_size
    )
    # dense tensors
    tokens_tensor = tf.feature_column.input_layer(
        features={'tokens': sequence['tokens']},
        feature_columns=[embedding_column]
    )
    labels_tensor = sequence['labels']
    lengths_tensor = context['length']
    return tokens_tensor, labels_tensor, lengths_tensor


def input_fn(data_path,
             filename,
             vocab_list,
             embedding_size,
             batch_size,
             num_epochs,
             sequence_length):
    file_path = path.join(data_path, filename)
    file_queue = tf.train.string_input_producer(
        [file_path],
        num_epochs=num_epochs
    )
    reader = tf.TFRecordReader()
    _, tfrecord = reader.read(file_queue)
    tokens, labels, length = parseExample(tfrecord, vocab_list, embedding_size)
    tokens_batch, labels_batch, length_batch = \
        tf.train.shuffle_batch(
            [tokens, labels, length],
            batch_size=batch_size,
            min_after_dequeue=1000,
            capacity=5000,
            allow_smaller_final_batch=True,
            shapes=[[sequence_length, embedding_size], [sequence_length], []]
        )
    features_batch = {
        'tokens': tokens_batch,
        'length': length_batch
    }
    return features_batch, labels_batch


def model_fn(features, labels, mode, params):
    if mode != tf.estimator.ModeKeys.TRAIN:
        params['dropout_keep_propd'] = 1.0

    with tf.name_scope('model'):
        cell_fw = tf.nn.rnn_cell.GRUCell(params['hidden_units'])
        cell_bw = tf.nn.rnn_cell.GRUCell(params['hidden_units'])
        _output = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw,
            features['tokens'],
            sequence_length=features['length'],
            dtype=tf.float32)
        ((output_fw, output_bw), _) = _output
        output = tf.concat([output_fw, output_bw], axis=-1)
        logits = tf.contrib.layers.fully_connected(
            output,
            params['num_classes'])

    with tf.name_scope('loss'):
        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits[:, :-1],
            labels=labels[:, 1:]
        )
        mask = tf.sequence_mask(features['length'])
        losses = tf.boolean_mask(cross_ent, mask)
        loss = tf.reduce_mean(losses)
        tf.summary.scalar('train loss', loss)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=params['learning_rate']
        )
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step()
        )

    with tf.name_scope('accuracy'):
        predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int64)
        correct_preds = tf.equal(predictions, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        tf.summary.scalar('train accuracy', accuracy)

    eval_metrics_ops = get_eval_metric_ops(labels, predictions)
    return EstimatorSpec(
        mode,
        train_op=train_op,
        predictions=predictions,
        loss=loss,
        eval_metric_ops=eval_metrics_ops)


def get_eval_metric_ops(labels, predictions):
    """Return a dict of the evaluation Ops.
    Args:
        labels (Tensor): Labels tensor for training and evaluation.
        predictions (Tensor): Predictions Tensor.
    Returns:
        Dict of metric results keyed by name.
    """
    return {
        'accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions,
            name='accuracy')
    }


def main(_):
    opt = {
        # Model options
        'embedding_size': 20,
        'rnn_units': 30,
        'learning_rate': 0.01,
        'batch_size': 32,
        'num_epochs': 10,
        'hidden_units': 30,
        'num_layers': 1,
        'dropout_keep_prob': 1.0,
        'sequence_length': 30,

        # Experiment options
        'save_summary_steps': 100,
        'save_checkpoints_steps': 100,
        'keep_checkpoint_max': 100,
        'log_step_count_steps': 100
    }
    data_path = path.join(path.dirname(__file__), '../output')
    model_dir = path.join(path.dirname(__file__), '../models')
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

    token_vocabs = read_vocab_list(path.join(data_path, 'tokens.vocab'))
    label_vocabs = read_vocab_list(path.join(data_path, 'labels.vocab'))
    opt['num_classes'] = len(label_vocabs)

    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(
        model_dir=model_dir,
        save_summary_steps=opt['save_summary_steps'],
        save_checkpoints_steps=opt['save_checkpoints_steps'],
        keep_checkpoint_max=opt['keep_checkpoint_max'],
        log_step_count_steps=opt['log_step_count_steps']
    )
    estimator = Estimator(
        model_fn=model_fn,
        params=opt,
        config=run_config
    )
    experiment = Experiment(
        estimator=estimator,
        train_input_fn=lambda: input_fn(
            data_path=data_path,
            filename='dev.tfrecords',
            batch_size=opt['batch_size'],
            num_epochs=opt['num_epochs'],
            vocab_list=token_vocabs,
            embedding_size=opt['embedding_size'],
            sequence_length=opt['sequence_length']
        ),
        eval_input_fn=lambda: input_fn(
            data_path=data_path,
            filename='test.tfrecords',
            batch_size=1,
            num_epochs=1,
            vocab_list=token_vocabs,
            embedding_size=opt['embedding_size'],
            sequence_length=opt['sequence_length']
        )
    )
    experiment.train_and_evaluate()


if __name__ == '__main__':
    main(sys.argv)
