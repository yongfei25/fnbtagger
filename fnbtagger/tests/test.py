from unittest import TestCase
from fnbtagger.generate_examples import (
     extract_labels, extract_tokens, TokenIndexer
)


class GenerateExampleTestCase(TestCase):
    def __init__(self, *args, **kwargs):
        super(GenerateExampleTestCase, self).__init__(*args, **kwargs)
        self.sentence = ('The/O quick/B-ORG brown/I-ORG fox/I-ORG jumps/O '
                         'over/O the/O lazy/B-PER dog/I-PER ./O')

    def test_extract_tokens(self):
        tokens = extract_tokens(self.sentence)
        expected = ['the', 'quick', 'brown', 'fox', 'jumps', 'over',
                    'the', 'lazy', 'dog', '.']
        self.assertEqual(tokens, expected)

    def test_extract_labels(self):
        tokens = extract_labels(self.sentence)
        expected = ['O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'B-PER',
                    'I-PER', 'O']
        self.assertEqual(tokens, expected)

    def test_token_indexer(self):
        indexer = TokenIndexer()
        result = indexer.index(self.sentence, extract_tokens)
        expected = [1, 2, 3, 4, 5, 6, 1, 7, 8, 9]
        self.assertEqual(result, expected)

    def test_labels_indexer(self):
        indexer = TokenIndexer()
        result = indexer.index(self.sentence, extract_labels)
        expected = [1, 2, 3, 3, 1, 1, 1, 4, 5, 1]
        self.assertEqual(result, expected)
