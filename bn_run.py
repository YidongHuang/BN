from scipy.io.arff import loadarff
import sys
from Naive_Bayes import Naive_Bayes
from TAN import TAN

def main(argv):
    train_file_data, train_file_meta = loadarff(argv[0])
    test_file_data, test_file_meta = loadarff(argv[1])
    option = argv[2]
    if option == 'n':
        nb = Naive_Bayes(train_file_data, train_file_meta, test_file_data)
        nb.test()
    if option == 't':
        tan = TAN(train_file_data, train_file_meta)


if __name__ == "__main__":
    main(sys.argv[1:])