from scipy.io.arff import loadarff
import sys
from Naive_Bayes import Naive_Bayes
from TAN import TAN
from random import random
import numpy as np

def main(argv):
    train_file_data, train_file_meta = loadarff(argv[0])
    test_file_data, test_file_meta = loadarff(argv[1])
    option = argv[2]
    if option == 'n':
        nb = Naive_Bayes(train_file_data, train_file_meta, test_file_data)
        nb.test()
    if option == 't':
        tan = TAN(train_file_data, train_file_meta, test_file_data)
        tan.test()
    # ten_fold_data, ten_fold_meta = loadarff('chess.arff')
    # test(ten_fold_data, ten_fold_meta)

def test(train_data, train_meta):
    test_blocks = get_test_blocks(train_data)
    nb_results = []
    tan_results = []
    for i in range(0, 10):
        test = test_blocks[i]
        train = []
        for j in range(0, 10):
            if j is not i:
                train += test_blocks[j]
        nb = Naive_Bayes(train, train_meta, test)
        nb_results.append(nb.test() * 1.0/len(test))
        tan = TAN(train, train_meta, test)
        tan_results.append(tan.test() * 1.0 /len(test))
    print [nb_results, tan_results]

def get_test_blocks(train_data):
    threshold = 0
    pos_samples = []
    neg_samples = []
    blocks = []
    for instance in train_data:
        if instance[-1] == 'won':
            threshold += 1
            pos_samples.append(instance)
        neg_samples.append(instance)
    threshold = threshold * 1.0/len(train_data)
    np.random.shuffle(pos_samples)
    np.random.shuffle(neg_samples)
    for i in range(0,10):
        block = []
        for j in range(0,  len(train_data)/10):
            if random() < threshold:
                if len(pos_samples) > 0:
                    block.append(pos_samples.pop())
            else:
                if len(neg_samples) > 0:
                    block.append(neg_samples.pop())
        blocks.append(block)
    return blocks

if __name__ == "__main__":
    main(sys.argv[1:])