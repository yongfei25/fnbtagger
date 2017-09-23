
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


def parseExample(record):
    # example feature
    context_features = {
        'length': tf.FixedLenFeature([], tf.int64)
    }
    sequence_features = {
        'tokens': tf.FixedLenSequenceFeature([], tf.int64),
        'labels': tf.FixedLenSequenceFeature([], tf.int64)
    }
    context, sequence = tf.parse_single_sequence_example(
        record,
        context_features=context_features,
        sequence_features=sequence_features
    )
    inputs = sequence['tokens']
    labels_tensor = sequence['labels']
    lengths_tensor = context['length']
    return inputs, labels_tensor, lengths_tensor


def input_fn(data_path,
             filename,
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
    tokens, labels, length = parseExample(tfrecord)
    tokens_batch, labels_batch, length_batch = \
        tf.train.shuffle_batch(
            [tokens, labels, length],
            batch_size=batch_size,
            min_after_dequeue=1000,
            capacity=5000,
            allow_smaller_final_batch=True,
            shapes=[[sequence_length], [sequence_length], []]
        )
    features_batch = {
        'tokens': tokens_batch,
        'length': length_batch
    }
    return features_batch, labels_batch


def create_model_fn(vocab_list, class_list):
    def model_fn(features, labels, mode, params):
        if params['num_layers'] < 1:
            raise ValueError('num_layers must be greater than or equal one.')
        if mode != tf.estimator.ModeKeys.TRAIN:
            params['dropout_keep_prob'] = 1.0

        with tf.variable_scope('inputs'):
            embeddings = tf.get_variable(
                name='embeddings',
                dtype=tf.float32,
                shape=[len(vocab_list), params['embedding_size']],
                initializer=tf.truncated_normal_initializer(
                    mean=0.0,
                    stddev=1.0/len(vocab_list)
                )
            )
            input_embeddings = tf.nn.embedding_lookup(embeddings, features['tokens'])
            inputs = tf.nn.dropout(input_embeddings, params['dropout_keep_prob'])
            sequence_length = tf.cast(features['length'], tf.int32)

        with tf.variable_scope('model'):
            cells_fw = []
            cells_bw = []
            for _ in range(params['num_layers']):
                cells_fw.append(tf.nn.rnn_cell.DropoutWrapper(
                    cell=tf.nn.rnn_cell.GRUCell(params['hidden_units']),
                    output_keep_prob=params['dropout_keep_prob']
                ))
                cells_bw.append(tf.nn.rnn_cell.DropoutWrapper(
                    cell=tf.nn.rnn_cell.GRUCell(params['hidden_units']),
                    output_keep_prob=params['dropout_keep_prob']
                ))
            _output = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.MultiRNNCell(cells_fw),
                cell_bw=tf.nn.rnn_cell.MultiRNNCell(cells_bw),
                inputs=inputs,
                sequence_length=sequence_length,
                dtype=tf.float32)
            ((output_fw, output_bw), _) = _output
            output = tf.concat([output_fw, output_bw], axis=-1)
            num_classes = len(class_list)
            logits = tf.contrib.layers.fully_connected(
                output,
                num_classes)
            predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {
                'classification': tf.estimator.export.ClassificationOutput(
                    scores=logits,
                    classes=tf.convert_to_tensor(class_list, tf.string)
                )
            }
            return EstimatorSpec(
                mode,
                predictions=predictions,
                export_outputs=export_outputs)
        else:
            labels = tf.cast(labels, tf.int32)
            with tf.variable_scope('loss'):
                cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=labels
                )
                mask = tf.sequence_mask(sequence_length)
                losses = tf.boolean_mask(cross_ent, mask)
                loss = tf.reduce_mean(losses)

                if mode == tf.estimator.ModeKeys.TRAIN:
                    tf.summary.scalar('train_loss', loss)

            with tf.variable_scope('train'):
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=params['learning_rate']
                )
                train_op = optimizer.minimize(
                    loss=loss, global_step=tf.train.get_global_step()
                )

            with tf.variable_scope('accuracy'):
                # NOTE: accuracy don't work well in this case,
                # Better option would be F1 score.
                if mode == tf.estimator.ModeKeys.TRAIN:
                    correct_preds = tf.equal(predictions, labels)
                    accuracy = tf.reduce_mean(
                        tf.cast(correct_preds, tf.float32))
                    tf.summary.scalar('train_accuracy', accuracy)

            eval_metrics_ops = {
                'accuracy': tf.metrics.accuracy(
                    labels=labels,
                    predictions=predictions,
                    name='accuracy')
            }
            
            return EstimatorSpec(
                mode,
                train_op=train_op,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metrics_ops)

    return model_fn


def create_experiment(user_opt={}):
    opt = {
        # Model options
        'embedding_size': 120,
        'hidden_units': 120,
        'learning_rate': 0.005,
        'batch_size': 256,
        'num_epochs': 50,
        'dropout_keep_prob': 1.0,
        'sequence_length': 30,
        'num_layers': 1,
        'random_seed': 1234,

        # Experiment options
        'save_summary_steps': 200,
        'save_checkpoints_steps': 500,
        'keep_checkpoint_max': 1,
        'log_step_count_steps': 100,
        'train_steps': None,

        # Paths
        'model_dir': path.join(path.dirname(__file__), '../models/default'),
        'data_dir': path.join(path.dirname(__file__), '../output'),
        'train_file': 'train.tfrecord'
    }
    opt.update(user_opt)
    data_dir = opt['data_dir']
    model_dir = opt['model_dir']
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

    token_vocabs = read_vocab_list(path.join(data_dir, 'tokens.vocab'))
    label_vocabs = read_vocab_list(path.join(data_dir, 'labels.vocab'))

    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(
        model_dir=model_dir,
        save_summary_steps=opt['save_summary_steps'],
        save_checkpoints_steps=opt['save_checkpoints_steps'],
        keep_checkpoint_max=opt['keep_checkpoint_max'],
        log_step_count_steps=opt['log_step_count_steps'],
        tf_random_seed=opt['random_seed']
    )
    estimator = Estimator(
        model_fn=create_model_fn(token_vocabs, label_vocabs),
        params=opt,
        config=run_config
    )
    experiment = Experiment(
        estimator=estimator,
        train_steps=opt['train_steps'],
        train_input_fn=lambda: input_fn(
            data_path=data_dir,
            filename=opt['train_file'],
            batch_size=opt['batch_size'],
            num_epochs=opt['num_epochs'],
            sequence_length=opt['sequence_length']
        ),
        eval_input_fn=lambda: input_fn(
            data_path=data_dir,
            filename='test.tfrecord',
            batch_size=1,
            num_epochs=1,
            sequence_length=opt['sequence_length']
        )
    )
    return experiment


def get_model_path(opt):
    keys = [
        'embedding_size', 'hidden_units', 'learning_rate',
        'dropout_keep_prob', 'num_epochs', 'num_layers'
    ]
    mPath = '-'.join(['{}={}'.format(x, opt[x]) for x in keys])
    return mPath
