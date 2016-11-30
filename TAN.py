import collections
import numpy as np
import math as math
import Node
class TAN:
    def __init__(self, train_data, train_meta, test_data):
        self.test_data = test_data
        self.train_data = train_data
        self.train_meta = train_meta
        self.feature_dict, self.feature_val_list, self.labels = self.get_feature_dict()
        self.matrix, self.count_labels = self.make_matrix()
        self.edges = self.get_edges()
        self.tree, self.y_node = self.get_TAN()
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
                edges[(x_feature, y_feature)] = self.compute_edge(x_feature, y_feature)
        return collections.OrderedDict(sorted(edges.items(), key=lambda t: t[1], reverse=True))

    def compute_edge(self, x_feature, y_feature):
        I = 0
        for z in (0, 1):
            for x_index in self.feature_dict[x_feature][1]:
                p_x_given_z = (self.matrix[z].sum(axis=1)[x_index] * 1.0 + 1)/ (len(self.feature_dict[x_feature][1]) + self.count_labels[z])
                print 'x feature {} index is {}, given z {} is {}'.format(x_feature, x_index, z, p_x_given_z)
                for y_index in self.feature_dict[y_feature][1]:
                    p_y_given_z = (self.matrix[z].sum(axis=1)[y_index] * 1.0 + 1)/(len(self.feature_dict[y_feature][1])+ self.count_labels[z])
                    p_x_y_given_z = (self.matrix[z][x_index, y_index] + 1) / (self.count_labels[z] + len(self.feature_dict[x_feature][1]) * len(self.feature_dict[y_feature][1]))
                    # if (x_index == 6 or 7) and (y_index == 12 or 13):
                    #     print self.matrix[z][x_index, y_index] + 1
                    #     print self.count_labels
                    #     print len(self.feature_dict[x_feature][1])
                    #     print len(self.feature_dict[x_feature][1])
                    #     print 'x feature {} index is {}, y feature {} index is{},  given z {} is {}'.format(x_feature, x_index, y_feature, y_index, z, p_x_y_given_z)
                    p_x_y_z = (self.matrix[z][x_index, y_index] + 1)/(self.count_labels[0] + self.count_labels[1] + len(self.feature_dict[x_feature][1]) * len(self.feature_dict[y_feature][1] * 2))
                    I += p_x_y_z * math.log(p_x_y_given_z/p_x_given_z/p_y_given_z)
        return I

    def get_max_spanning_tree(self):
        tree = {}
        avail_nodes = {}
        for i in range(len(self.train_meta.names())):
            name = self.train_meta.names()[i]
            node = Node.Node(name)
            if i == 0:
                node.key_val = 0
                tree[name] = node
                continue
            if name == 'class':
                continue
            avail_nodes[name] = node

        while len(avail_nodes) > 0:
            from_name, to_name = self.get_max_distances(tree.keys(), avail_nodes.keys())
            to_node = avail_nodes.pop(to_name)
            tree[from_name].children.append(to_node)
            tree[to_name] = to_node
        return tree


    def get_max_distances(self, unavail_names, avail_names):
        unavail_names = sorted(unavail_names, key=lambda name: self.train_meta.names().index(name))
        avail_names = sorted(avail_names, key=lambda name: self.train_meta.names().index(name))
        min_dist = -1
        chosen_dist = None
        chosen_origin = None
        for unavail_name in unavail_names:
            for avail_name in avail_names:
                if self.edges[self.get_distance_key(unavail_name, avail_name)] > min_dist:
                    min_dist = self.edges[self.get_distance_key(unavail_name, avail_name)]
                    chosen_dist = avail_name
                    chosen_origin = unavail_name
        return chosen_origin, chosen_dist

    def get_distance_key(self, x, y):
        if self.train_meta.names().index(x) > self.train_meta.names().index(y):
            return (y,x)
        return (x,y)

    def get_TAN(self):
        tree = self.get_max_spanning_tree()
        for node in tree.values():
            print node.name
            print node.children
        tree_node = Node.Node('class')
        tree_node.children = tree.values()
        return tree, tree_node


    def get_feature_dict(self):
        feature_val_list = []
        index_start = 0
        feature_dict = {}
        labels = []
        for name in self.train_meta.names():
            if name == 'class':
                labels = self.train_meta['class'][1]
                break
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
        num_pos = 0
        neg_matrix = np.zeros((len(self.feature_val_list), len(self.feature_val_list)))
        matrix = [pos_matrix, neg_matrix]
        for instance in self.train_data:
            index = 1
            if instance[-1] == self.labels[0]:
                index = 0
                num_pos += 1
            for i in range(len(instance) - 2):
                x_val = instance[i]
                feature_name_x = self.train_meta.names()[i]
                feature_x = self.feature_dict[feature_name_x]
                x_index = feature_x[1][feature_x[0].index(x_val)]
                for j in range(i + 1, len(instance) - 1):
                    y_val = instance[j]
                    feature_name_y = self.train_meta.names()[j]
                    feature_y = self.feature_dict[feature_name_y]
                    y_index = feature_y[1][feature_y[0].index(y_val)]
                    matrix[index][x_index, y_index] += 1
        return matrix, [num_pos, len(self.train_data) - num_pos]



