# fnbtagger

A naive LSTM implementation in Tensorflow to tag words related to food and beverages.

Demo: http://fer.kaover.com/

## Scripts
- `generate_example.py` - Generate tfrecord files from raw data files.
- `tune/step_1.py` - Hyperparameters tuning with random search.
- `export_model.py` - Export SavedModel from checkpoint.
