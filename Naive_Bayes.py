import collections
class Naive_Bayes:
    def __init__(self, train_data, train_meta, test_data):
        self.test_data = test_data
        self.labels, self.train_dict, self.feature_dict = self.get_train_dict(train_meta)
        self.train_size = len(train_data)
        self.num_positive, self.positive_probability = self.train(train_data)
        # print self.labels
        # print self.train_dict
        # print self.feature_dict



    def get_train_dict(self, train_meta):
        train_dict = collections.OrderedDict()
        feature_dict = collections.OrderedDict()
        labels = []
        for i in range(len(train_meta.names())):
            name = train_meta.names()[i]
            if not name == train_meta.names()[-1]:
                train_dict[name] = collections.OrderedDict()
                feature_dict[i] = name
                for feature in train_meta[name][1]:
                    train_dict[name][feature] = [0, 0]
            else:
                labels.append(train_meta[name][1][0])
                labels.append(train_meta[name][1][1])
        return labels, train_dict, feature_dict

    def train(self, train_data):
        num_positive = 0
        for instance in train_data:
            label_index = 1
            if instance[-1] == self.labels[0]:
                label_index = 0
                num_positive += 1
            for i in range(len(instance) - 1):
                feature_name = self.feature_dict[i]
                self.train_dict[feature_name][instance[i]][label_index] += 1
        return num_positive, (num_positive * 1.0 + 1)/(self.train_size + 2)

    def test(self):
        for i in range(len(self.test_data[0]) - 1):
            print '{} class'.format(self.feature_dict[i])
        print '\r'
        correct_prediction = 0
        for instance in self.test_data:
            pos_probability = self.positive_probability
            neg_probability = 1.0 * (self.train_size - self.num_positive + 1)/(self.train_size + 2)
            for i in range(len(instance) - 1):
                instance_dict = self.train_dict[self.feature_dict[i]][instance[i]]
                pos_probability *= (instance_dict[0] * 1.0 + 1)/ (self.num_positive + len(self.train_dict[self.feature_dict[i]]))
                neg_probability *= (instance_dict[1] * 1.0 + 1) /(self.train_size - self.num_positive  + len(self.train_dict[self.feature_dict[i]]))
            probability = (pos_probability)/(pos_probability + neg_probability)
            if probability >= 0.5:
                prediction = self.labels[0]
            else:
                prediction = self.labels[1]
                probability = 1- probability
            print '{0} {1} {2:.12f}'.format(prediction.replace("'",""), instance[-1].replace("'",""), probability)
            if prediction == instance[-1]:
                correct_prediction += 1

        print '\r'
        print correct_prediction
        return correct_prediction


