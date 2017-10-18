import tensorflow as tf
from fnbtagger.generate_example import make_example


class GenerateExampleV2Test(tf.test.TestCase):
    def test_make_example(self):
        with self.test_session():
            ex = make_example([0, 1], [0, 1])
            features = tf.parse_single_sequence_example(
                ex.SerializeToString(),
                {'length': tf.FixedLenFeature(
                    [],
                    tf.int64)},
                {'tokens': tf.FixedLenSequenceFeature(
                    [],
                    tf.int64),
                 'labels': tf.FixedLenSequenceFeature(
                    [],
                    tf.int64)}
            )
            context, sequences = features
            self.assertAllEqual(context['length'].eval(), 2)
            self.assertAllEqual(sequences['tokens'].eval(), [0, 1])
            self.assertAllEqual(sequences['labels'].eval(), [0, 1])


if __name__ == '__main__':
    tf.test.main()
