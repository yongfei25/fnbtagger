import argparse
import shutil
from os import path
from fnbtagger.train_lib import create_experiment


def main(lang):
    data_dir = path.join(path.dirname(__file__), '../output/{}'.format(lang))
    model_dir = path.join(path.dirname(__file__),
                          '../models/{}/dev'.format(lang))
    shutil.rmtree(model_dir, ignore_errors=True)
    opt = {
        'data_dir': data_dir,
        'model_dir': model_dir,
        'embedding_size': 250,
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
    parser = argparse.ArgumentParser(description='Generate TFrecord files.')
    parser.add_argument('language', choices=['en', 'zh'])
    args = parser.parse_args()
    main(args.language)
