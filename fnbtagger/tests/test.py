from unittest import TestCase

import fnbtagger

class FirstTest(TestCase):
  def test_is_string(self):
    result = fnbtagger.hello()
    self.assertTrue(isinstance(result, str))
