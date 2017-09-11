import os
import tensorflow as tf
from fnbtagger.generate_example_v2 import make_example


class GenerateExampleV2Test(tf.test.TestCase):
    def test_make_example(self):
        with self.test_session():
            ex = make_example(['good', 'day'], ['O', 'P'])
            features = tf.parse_single_sequence_example(
                ex.SerializeToString(),
                {'length': tf.FixedLenFeature(
                    [],
                    tf.int64)},
                {'tokens': tf.FixedLenSequenceFeature(
                    [],
                    tf.string),
                 'labels': tf.FixedLenSequenceFeature(
                    [],
                    tf.string)}
            )
            context, sequences = features
            self.assertAllEqual(context['length'].eval(), 2)
            self.assertAllEqual(sequences['tokens'].eval(), [b'good', b'day'])
            self.assertAllEqual(sequences['labels'].eval(), [b'O', b'P'])


if __name__ == '__main__':
    tf.test.main()
