import collections
from copy import deepcopy
class TAN:
    def __init__(self, train_data, train_meta, test_data):
        self.test_data = test_data
        self.train_data = train_data
        self.train_meta = train_meta
        self.feature_dict = self.get_feature_dict()
        self.pre_processed_data = self.construct_dict()
        self.pre_process()
        self.edges = self.get_edges()
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
        pos = self.pre_processed_data[0]
        pos_p_x_y_z = 0
        pos_x_feature = pos[1][x_feature]
        for key in pos_x_feature[1].keys():
            x_val = pos_x_feature[1][key]
            p_x_z = (x_val[0] + 1)/(pos[1] + 1)
        neg = self.pre_processed_data[1]


    # def get_feature_dict(self):
    #     feature_list = []
    #     feature_dict = collections.OrderedDict()
    #     for name in self.train_meta.names():
    #         if name == 'class':
    #             break
    #         feature_dict[name] = self.train_meta[name][1]
    #     return feature_dict

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



