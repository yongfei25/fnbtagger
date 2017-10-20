import argparse
import os
import random
from os import path
from fnbtagger.train_lib import create_experiment, get_model_path


def main(lang):
    """
    Search these parameters:
        'embedding_size'
        'hidden_units'
        'learning_rate'
        'dropout_keep_prob'
        'num_layers'
    """
    data_dir = path.join(path.dirname(__file__),
                         '../../output/{}'.format(lang))
    model_dir = path.join(path.dirname(__file__),
                          '../../models/{}/step-1'.format(lang))
    iterations = 50
    default = {
        'data_dir': data_dir,
        'num_epochs': 50,
        'train_file': 'dev.tfrecord'
    }
    ran = []
    if os.path.isdir(model_dir):
        ran = next(os.walk(model_dir))[1]

    for _ in range(iterations):
        opt = {}
        opt.update(default)
        opt['dropout_keep_prob'] = random.choice([0.7, 0.5, 0.4, 0.2])
        opt['hidden_units'] = random.choice([20, 30, 50, 100, 150, 200])
        opt['embedding_size'] = random.choice([20, 30, 50, 100, 150, 200])
        opt['learning_rate'] = random.choice([0.0005, 0.001, 0.003])
        opt['num_layers'] = random.choice([1])
        model_path = get_model_path(opt)
        if model_path in ran:
            continue
        opt['model_dir'] = path.join(model_dir, model_path)
        print("Tune: " + model_path)
        ran.append(model_path)
        experiment = create_experiment(opt)
        experiment.train_and_evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TFrecord files.')
    parser.add_argument('language', choices=['en', 'zh'])
    args = parser.parse_args()
    main(args.language)
