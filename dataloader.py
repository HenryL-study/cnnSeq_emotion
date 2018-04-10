import numpy as np

class Dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.MAX_SEQ_LEN = 100 #change every time

    def load_train_data(self, positive_file, positive_len_file, negative_file, negative_len_file):
        # Load data
        positive_examples = []
        negative_examples = []
        positive_examples_len = []
        negative_examples_len = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                positive_examples.append(parse_line)
        with open(positive_len_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                positive_examples_len = [int(x) for x in line]        
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                negative_examples.append(parse_line)
        with open(negative_len_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                negative_examples_len = [int(x) for x in line] 

        self.sentences = np.array(positive_examples + negative_examples)
        self.sentences_len = np.concatenate((np.array(positive_examples_len), np.array(negative_examples_len)))

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]
        self.sentences_len = self.sentences_len[shuffle_indices]

        # Split batches
        self.total_batch = int(len(self.labels) / self.batch_size)

        self.num_test_batch = int(self.total_batch * 0.2)
        self.num_train_batch = self.total_batch - self.num_test_batch


        self.sentences = self.sentences[:self.total_batch * self.batch_size]
        self.labels = self.labels[:self.total_batch * self.batch_size]
        self.sentences_len = self.sentences_len[:self.total_batch * self.batch_size]

        self.sentences_batches = np.split(self.sentences, self.total_batch, 0)
        self.labels_batches = np.split(self.labels, self.total_batch, 0)
        self.sequences_len_batches = np.split(self.sentences_len, self.total_batch)

        self.pointer = 0
        self.test_pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer]
        seq_len = self.sequences_len_batches[self.pointer]
        label = self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_train_batch
        return ret, seq_len, label
    
    def next_test_batch(self):
        ret = self.sentences_batches[self.test_pointer + self.num_train_batch]
        seq_len = self.sequences_len_batches[self.test_pointer + self.num_train_batch]
        label = self.labels_batches[self.test_pointer + self.num_train_batch]
        self.test_pointer = (self.test_pointer + 1) % self.num_test_batch
        return ret, seq_len, label

    def reset_pointer(self):
        self.pointer = 0
        self.test_pointer = 0

