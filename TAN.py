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
        self.edge_matrix = np.zeros((len(self.train_meta.names()) - 1, len(self.train_meta.names()) - 1))
        self.matrix, self.count_labels, self.prob_table = self.make_matrix()
        self.edges = self.get_edges()
        self.tree, self.y_node = self.get_TAN()
        self.table = self.make_conditional_prob_table()

    def get_edges(self):
        edges = collections.OrderedDict()
        for i in range(len(self.train_meta.names()) - 2):
            x_feature = self.train_meta.names()[i]
            for j in range(i+1, len(self.train_meta.names()) - 1):
                y_feature = self.train_meta.names()[j]
                edges[(x_feature, y_feature)] = self.compute_edge(x_feature, y_feature)
                self.edge_matrix[i, j] = edges[(x_feature, y_feature)]
        return collections.OrderedDict(sorted(edges.items(), key=lambda t: t[1], reverse=True))

    def compute_edge(self, x_feature, y_feature):
        I = 0
        for z in (0, 1):
            for x_index in self.feature_dict[x_feature][1]:
                p_x_given_z = (self.prob_table[z][x_index]* 1.0 + 1)/ (len(self.feature_dict[x_feature][1]) + self.count_labels[z])
                for y_index in self.feature_dict[y_feature][1]:
                    p_y_given_z = (self.prob_table[z][y_index]* 1.0 + 1)/(len(self.feature_dict[y_feature][1])+ self.count_labels[z])
                    p_x_y_given_z = (self.matrix[z][x_index, y_index] + 1) / (self.count_labels[z] + len(self.feature_dict[x_feature][1]) * len(self.feature_dict[y_feature][1]))
                    p_x_y_z = (self.matrix[z][x_index, y_index] + 1)/(len(self.train_data) + len(self.feature_dict[x_feature][1]) * len(self.feature_dict[y_feature][1]) * 2)
                    # p_x_y_z = (self.matrix[z][x_index, y_index] + 1) / (len(self.train_data) )
                    I += p_x_y_z * math.log(p_x_y_given_z/p_x_given_z/p_y_given_z, 2)
        return I

    def get_max_spanning_tree(self):
        tree = {}
        avail_nodes = {}
        for i in range(len(self.train_meta.names())):
            name = self.train_meta.names()[i]
            node = Node.Node(name)
            if i == 0:
                tree[name] = node
                node.parent_name = 'class'
                continue
            if name == 'class':
                continue
            avail_nodes[name] = node

        while len(avail_nodes) > 0:
            from_name, to_name = self.get_max_distances(tree.keys(), avail_nodes.keys())
            to_node = avail_nodes.pop(to_name)
            tree[from_name].children.append(to_node)
            tree[to_name] = to_node
            to_node.parent_name = from_name
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
        tree_node = Node.Node('class')
        tree_node.children = tree.values()
        print '{} {}'.format(self.train_meta.names()[0], 'class')
        for i in range(1, len(self.train_meta.names()) - 1):
            node = tree[self.train_meta.names()[i]]
            print '{} {} class'.format(node.name, node.parent_name)
        print '\r'
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

    def make_matrix(self):
        pos_matrix = np.zeros((len(self.feature_val_list), len(self.feature_val_list)))
        num_pos = 0
        neg_matrix = np.zeros((len(self.feature_val_list), len(self.feature_val_list)))
        prob_table = [len(self.feature_val_list) * [0], len(self.feature_val_list) * [0]]
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
            for k in range(len(instance) - 1):
                prob_table[index][self.feature_val_list.index(self.train_meta.names()[k] + '_' + instance[k])] += 1
        return matrix, [num_pos, len(self.train_data) - num_pos], prob_table

    def make_conditional_prob_table(self):
        conditional_prob_dict = {}
        for node in self.tree.values():
            if node.name is not self.train_meta.names()[0]:
                conditional_prob_dict[(node.name, node.parent_name)] = self.compute_feature_conditional_prob(node.name, node.parent_name)
        return conditional_prob_dict

    def compute_feature_conditional_prob(self, node_name, parent_name):
        feature_conditional_dict = {}
        for x_val in self.train_meta[node_name][1]:
            for y_val in self.train_meta[parent_name][1]:
                for z_val in self.labels:
                    z_index = 1
                    if z_val == self.labels[0]:
                        z_index = 0
                    feature_conditional_dict[(x_val, y_val, z_val)] = self.compute_conditional_prob(self.feature_val_list.index(node_name + '_' + x_val),self.feature_val_list.index(parent_name + '_' + y_val), z_index, node_name, parent_name)
        return feature_conditional_dict

    def compute_conditional_prob(self, x_index, y_index, z_index, x_feature_name, y_feature_name):
        matrix = self.matrix[z_index]
        prob_table = self.prob_table[z_index]
        parent_count = prob_table[y_index]
        if x_index < y_index:
            # conditional_prob = (matrix[x_index, y_index] + 1) * 1.0/(parent_count + len(self.feature_dict[y_feature_name][1]))
            conditional_prob = (matrix[x_index, y_index] + 1) * 1.0 / (parent_count + len(self.feature_dict[x_feature_name][1]))
        else:
            # conditional_prob = (matrix[y_index, x_index] + 1) * 1.0/(parent_count + len(self.feature_dict[y_feature_name][1]))
            conditional_prob = (matrix[y_index, x_index] + 1) * 1.0 / (parent_count + len(self.feature_dict[x_feature_name][1]))
        # print 'x is {}, y is {}, given z is {}, prob is {}'.format(x_feature_name, y_feature_name, z_index, conditional_prob)
        return conditional_prob

    def test(self):
        correct_prediction = 0
        for instance in self.test_data:
            pos_index = 0
            pos_p_condition = (self.count_labels[pos_index] * 1.0 + 1) / (len(self.train_data) + 2)
            # pos_p_root_given_z = (self.prob_table[pos_index][self.feature_val_list.index(self.train_meta.names()[0] + '_' + instance[0])] * 1.0 + 1) / (self.count_labels[pos_index] + 2)
            pos_p_root_given_z = (self.prob_table[pos_index][self.feature_val_list.index(self.train_meta.names()[0] + '_' + instance[0])] *1.0 + 1)/(self.count_labels[pos_index] + len(self.train_meta[self.train_meta.names()[0]][1]))
            pos_p = pos_p_condition * pos_p_root_given_z
            for i in range(1, len(instance) - 1):
                node = self.tree[self.train_meta.names()[i]]
                parent_index = self.train_meta.names().index(node.parent_name)
                feature_conditional_dict = self.table[(node.name, node.parent_name)]
                conditional_p = feature_conditional_dict[(instance[i], instance[parent_index], self.labels[pos_index])]
                pos_p *= conditional_p
            pos_index = 1
            neg_p_condition = (self.count_labels[pos_index] * 1.0 + 1) / (len(self.train_data) + 2)
            # neg_p_root_given_z = (self.prob_table[pos_index][self.feature_val_list.index(self.train_meta.names()[0] + '_' + instance[0])] * 1.0 + 1) / (self.count_labels[pos_index] + 2)
            neg_p_root_given_z = (self.prob_table[pos_index][self.feature_val_list.index(self.train_meta.names()[0] + '_' + instance[0])] *1.0 + 1)/(self.count_labels[pos_index] + len(self.train_meta[self.train_meta.names()[0]][1]))
            neg_p = neg_p_condition * neg_p_root_given_z
            for i in range(1, len(instance) - 1):
                node = self.tree[self.train_meta.names()[i]]
                parent_index = self.train_meta.names().index(node.parent_name)
                feature_conditional_dict = self.table[(node.name, node.parent_name)]
                conditional_p = feature_conditional_dict[(instance[i], instance[parent_index], self.labels[pos_index])]
                neg_p *= conditional_p
            p = pos_p/(pos_p + neg_p)
            if p > 0.5:
                prediction = self.labels[0]
            else:
                prediction = self.labels[1]
                p = 1-p
            print '{0} {1} {2:.12f}'.format(prediction.replace("'", ""), instance[-1].replace("'",""), p)
            if prediction == instance[-1]:
                correct_prediction += 1
        print '\r'
        print correct_prediction
        return correct_prediction

