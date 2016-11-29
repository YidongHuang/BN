import collections
import numpy as np
from copy import deepcopy
class TAN:
    def __init__(self, train_data, train_meta, test_data):
        self.test_data = test_data
        self.train_data = train_data
        self.train_meta = train_meta
        self.feature_dict, self.feature_val_list, self.labels = self.get_feature_dict()
        self.matrix = self.make_matrix()
        # self.pre_processed_data = self.construct_dict()
        # self.pre_process()
        # print self.feature_val_list
        # print "************"
        # print self.feature_dict
        # self.edges = self.get_edges()

        # print self.feature_dict
        # print "***************"
        # print self.pre_processed_data[0]

    # def pre_process(self):
    #     class_labels = self.train_meta['class'][1]
    #     pos_label = class_labels[0]
    #     for instance in self.train_data:
    #         index = 1
    #         if instance[-1] == pos_label:
    #             index = 0
    #         classified_data = self.pre_processed_data[index]
    #         classified_data[0] += 1
    #         for i in range(len(instance) - 2):
    #             x_val = instance[i]
    #             given_x_feature = classified_data[1][self.train_meta.names()[i]]
    #             given_x_feature[0] += 1
    #             given_x_val = given_x_feature[1][x_val]
    #             given_x_val[0] += 1
    #             for j in range(i+1, len(instance) - 1):
    #                 y_val = instance[j]
    #                 given_y_feature = given_x_val[1][self.train_meta.names()[j]]
    #                 given_y_feature[0] += 1
    #                 given_y_feature[1][y_val] += 1
    #     return

    def get_edges(self):
        edges = collections.OrderedDict()
        for i in range(len(self.train_meta.names()) - 2):
            x_feature = self.train_meta.names()[i]
            for j in range(i+1, len(self.train_meta.names()) - 1):
                y_feature = self.train_meta.names()[j]
                edges[(x_feature, y_feature)] = self.computer_edge(x_feature, y_feature)
        return edges

    def compute_edges(self, x_feature, y_feature):
        for z in (0, 1):
            z_sum = self.matrix[z].sum()
            for x_index in self.feature_dict[x_feature][1]:
                p_x_given_z = (self.matrix[z].sum(axis=1)[x_index] * 1.0 + 1)/ (len(self.feature_dict[x_feature][1]) + z_sum)
                for y_index in self.feature_dict[y_feature][1]:
                    p_y_given_z = (self.matrix[z].sum(axis=2)[y_index] * 1.0 + 1)/(len(self.feature_dict[y_feature][1])+ z_sum)
                    P_x_y_given_z = self.matrix[z][x_index, y_index]



    def get_feature_dict(self):
        feature_val_list = []
        index_start = 0
        feature_dict = {}
        labels = []
        for name in self.train_meta.names():
            if name == 'class':
                labels = self.train_meta['class'][1]
            feature_dict[name] = [self.train_meta[name][1], range(index_start, index_start + len(self.train_meta[name][1]))]
            index_start += len(self.train_meta[name][1])
            feature_val_list.extend([name + "_"+ val for val in self.train_meta[name][1]])
        return feature_dict, feature_val_list, labels

    # def construct_dict(self):
    #     all_feature_combination = collections.OrderedDict()
    #     for i in range(len(self.train_meta.names()) - 2):
    #         x_feature_dict = collections.OrderedDict()
    #         x_feature_name = self.train_meta.names()[i]
    #         x_features = self.feature_dict[self.train_meta.names()[i]]
    #         for j in range(len(x_features)):
    #             x_val = x_features[j]
    #             y_feature_dict = collections.OrderedDict()
    #             for k in range(i+1, len(self.train_meta.names()) - 1):
    #                 xy_combination = collections.OrderedDict()
    #                 y_feature_name = self.train_meta.names()[k]
    #                 y_features = self.feature_dict[self.train_meta.names()[k]]
    #                 for m in range(len(y_features)):
    #                     xy_combination[y_features[m]] = 0
    #                 y_feature_dict[y_feature_name] = [0,xy_combination]
    #             x_feature_dict[x_val] = [0, y_feature_dict]
    #         all_feature_combination[x_feature_name] = [0, x_feature_dict]
    #     cpy_all_feature_combination = deepcopy(all_feature_combination)
    #     return [[0,all_feature_combination], [0,cpy_all_feature_combination]]

    def make_matrix(self):
        pos_matrix = np.zeros((len(self.feature_val_list), len(self.feature_val_list)))
        neg_matrix = np.zeros((len(self.feature_val_list), len(self.feature_val_list)))
        matrix = [pos_matrix, neg_matrix]
        for instance in self.train_data:
            index = 1
            if instance[-1] == self.labels[0]:
                index = 0
            for i in range(len(instance) - 2):
                x_val = instance[i]
                feature_name_x = self.train_meta.names()[i]
                feature_x = self.feature_dict[feature_name_x]
                x_index = feature_x[1][feature_x[0].index(x_val)]
                for j in range(i+1, len(instance) - 1):
                    y_val = instance[j]
                    feature_name_y = self.train_meta.names()[j]
                    feature_y = self.feature_dict[feature_name_y]
                    y_index = feature_y[1][feature_y[0].index(y_val)]
                    matrix[index][x_index, y_index] += 1
        return matrix



