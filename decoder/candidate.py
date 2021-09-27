class Candidate:
    def __init__(self, previous_node, token, length):
        self.previous_node = previous_node
        self.token = token # can be [SEP] (str)
        self.length = length
        self.similarity = 0
        self.fluency = 0
        self.depth = 1
        self.text = None
        self.sequence = []
        
    
    def __str__(self):
        msg = f'- Word: {self.word}\n- Length: {self.length}\n- Similarity: {self.similarity}'
        return msg
    
    def func(self, sequences):
        sequences.append(self.token)
        if self.previous_node is None:
            return sequences
        else: 
            return self.previous_node.func(sequences)
    
    def reverse(self):
        sequences = []
        self.sequences = self.func(sequences)