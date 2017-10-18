class DatasetSplitter:
    def __init__(self, split=0.9):
        if split <= 0 or split >= 1:
            raise ValueError('The split must beween 0 and 1')
        self.split_a = split
        self.split_b = 1 - split
        self.allocation = {
            'set_a': 0,
            'set_b': 0,
            'total': 0
        }

    def allocate(self):
        total = self.allocation['total'] + 1
        target = None
        if total == 0:
            target = 'set_a'
        elif self.allocation['set_a'] / total < self.split_a:
            target = 'set_a'
        else:
            target = 'set_b'
        self.allocation[target] += 1
        self.allocation['total'] = total
        return target


class TokenIndexer:
    def __init__(self, max_length=51, unk='<unk>', pad='<pad>'):
        self.ids = {}
        self.tokens = {}
        self.ids[pad] = 0
        self.tokens[0] = pad
        self.ids[unk] = 1
        self.tokens[1] = unk
        self.unk = unk
        self.pad = pad
        self.last_id = 1
        self.max_length = max_length

    def index(self, sentence, extract_func):
        tokens = extract_func(sentence)
        return self.index_tokens(tokens)

    def index_tokens(self, tokens):
        indexes = []
        for token in tokens:
            if token in self.ids:
                indexes.append(self.ids[token])
            else:
                self.last_id += 1
                self.ids[token] = self.last_id
                self.tokens[self.last_id] = token
                indexes.append(self.last_id)
        indexes = self.pad_right(indexes, self.max_length, self.ids[self.pad])
        return indexes

    def get_ids(self, tokens):
        indexes = [self.ids.get(token, self.ids[self.unk]) for token in tokens]
        return self.pad_right(indexes, self.max_length, self.ids[self.pad])

    def get_tokens(self, ids):
        indexes = [self.tokens.get(tId, self.tokens[1]) for tId in ids]
        return self.pad_right(indexes, self.max_length, self.tokens[0])

    def pad_right(self, aList, length, padding):
        cList = list(aList)
        l = len(aList)
        while l < length:
            cList.append(padding)
            l += 1
        return cList
