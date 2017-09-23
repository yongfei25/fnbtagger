import sys
import shutil
from os import path
from fnbtagger.train_lib import create_experiment


def main(_):
    data_dir = path.join(path.dirname(__file__), '../output')
    model_dir = path.join(path.dirname(__file__), '../models-dev/')
    shutil.rmtree(model_dir, ignore_errors=True)
    opt = {
        'data_dir': data_dir,
        'model_dir': model_dir,
        'embedding_size': 100,
        'hidden_units': 100,
        'learning_rate': 0.001,
        'save_summary_steps': 1,
        'dropout_keep_prob': 1.0,
        'num_epochs': 50,
        'num_layers': 1,
        'train_file': 'tiny.tfrecord',
        'batch_size': 1
    }
    experiment = create_experiment(opt)
    experiment.train()

if __name__ == '__main__':
    main(sys.argv)
