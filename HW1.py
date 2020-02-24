# Programmer : Zeid Al-Ameedi
# Dates : 1-23-2020 -> 1-27-2020
# Collab : Diane Cook, stackoverflow and https://medium.com/@rishabhjain_22692/decision-trees-it-begins-here-93ff54ef134
# https://medium.com/@pytholabs/decision-trees-from-scratch-using-id3-python-coding-it-up-6b79e3458de4


import collections
import numpy
import operator
#from google.colab import drive
#from google.colab import files
import math

#drive.mount('/content/gdrive')

"""
Tree Node class - was trying to insert the entire decision tree in a more
familar data structure in order to print it out at the end.
"""


class TreeNode:
    def __init__(self, val):
        self.l_child = None
        self.r_child = None
        self.data = val

    def binary_insert(self, root, node):
        if root is None:
            root = node
        else:
            if root.data > node.data:
                if root.l_child is None:
                    root.l_child = node
                else:
                    self.binary_insert(root.l_child, node)
            else:
                if root.r_child is None:
                    root.r_child = node
                else:
                    self.binary_insert(root.r_child, node)

    def in_order_print(self, root):
        if not root:
            return
        self.in_order_print(root.l_child)
        print(root.data)
        self.in_order_print(root.r_child)

    def pre_order_print(self, root):
        if not root:
            return
        print(root.data)
        self.pre_order_print(root.l_child)
        self.pre_order_print(root.r_child)


def report_statistics(total, tp, fp, tn, fn):
    print("total", total, "tp", tp, "fp", fp, "tn", tn, "fn", fn)


# Classify test instances based on majority label
def simple_majority_test(X, y, majority_class):
    total = len(y)
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i in range(total):  # evaluate each test instance
        label = majority_class  # not really needed, just illustrates point
        if label == 0.0:  # majority label is negative
            if y[i] == 0.0:  # this is a negative instance
                true_negative += 1
            else:  # this is a positive instance
                false_negative += 1
        else:  # majority label is positive (label == 1.0)
            if y[i] == 0.0:  # this is a negative instance
                false_positive += 1
            else:  # this is a positive instance
                true_positive += 1
    report_statistics(total, true_positive, false_positive,
                      true_negative, false_negative)


ft = ["color", "type", "origin"]

color_metrics = ["red", "yellow"]

type_metrics = ["sports", "suv"]

origin_metrics = ["domestic", "imported"]

metrics = [color_metrics, type_metrics, origin_metrics]

# Hard coded because numpy arrays are terrible and i got CONFUSED reading them dynamically into our tree
trainData = {1: {"color": "red", "type": "sports", "origin": "domestic", "stolen": "yes"},
             2: {"color": "red", "type": "sports", "origin": "domestic", "stolen": "no"},
             3: {"color": "red", "type": "sports", "origin": "domestic", "stolen": "yes"},
             4: {"color": "yellow", "type": "sports", "origin": "domestic", "stolen": "no"},
             5: {"color": "yellow", "type": "sports", "origin": "imported", "stolen": "yes"},
             6: {"color": "yellow", "type": "suv", "origin": "imported", "stolen": "no"},
             7: {"color": "yellow", "type": "suv", "origin": "imported", "stolen": "yes"},
             8: {"color": "yellow", "type": "suv", "origin": "domestic", "stolen": "no"},
             9: {"color": "red", "type": "suv", "origin": "imported", "stolen": "no"},
             10: {"color": "red", "type": "sports", "origin": "imported", "stolen": "yes"}}


# def split(node, max_depth, depth):
#    left, right = node['groups']
#    del(node['groups'])
#    if not left or not right:
#       node['left'] = node['right'] = create_leaf(left + right)
#       return
#    if depth >= max_depth:
#       node['left'], node['right'] = create_leaf(left), create_leaf(right)
#       return
#    node['left'] = select_attribute(left)
#    split(node['left'], max_depth, depth+1)
#    node['right'] = select_attribute(right)
#    split(node['right'], max_depth, depth+1)

class Node(object):
    def __init__(self, name, features=ft, parent=None):
        self.name = name
        self.branches = []
        self.features = features
        self.entropy = 0.0
        self.parent = parent
        self.left = None
        self.right = None
        self.stolen = None

    def set_entropy(self, entropy):
        self.entropy = entropy

    def set_stolen(self, stolen):
        self.stolen = stolen

    def buildaBranch(self, name):
        self.branches.append(name)

    # Debug function used on node object
    def isValid(self, name):
        if self.name == name:
            print(self.name)
        else:
            print(False)

    def cleanBranch(self, b):
        self.features.remove(b)

    def default(self):
        self.features = ["color", "type", "origin"]


def EntropyFormula(entropyAttributes):
    true_Stolen = 0
    false_stolen = 0
    total = 0
    index = 0
    if (len(entropyAttributes) == 0):
        for id in trainData:
            if (trainData[id]["stolen"] == "yes"):
                true_Stolen += 1
            else:
                false_stolen += 1
            total += 1
    else:
        for id in trainData:
            for attr in entropyAttributes:
                if (attr in trainData[id].values()):
                    index += 1
            if (index == len(entropyAttributes)):
                if (trainData[id]["stolen"] == "yes"):
                    true_Stolen += 1
                else:
                    false_stolen += 1
                total += 1
            index = 0
    return [round((-(true_Stolen / total) * math.log((true_Stolen / total), 2)) - (
            (false_stolen / total) * math.log((false_stolen / total), 2)), 3), true_Stolen,
            false_stolen, total]


def load_data(ty):
    if ty == 'train':
        data = numpy.loadtxt(fname='/content/gdrive/My Drive/content/TrainData.csv', dtype='str', delimiter=',',
                             encoding='utf-8-sig')
        X = data[:, :-1]
        y = data[:, -1]
        return X, y
    else:
        data = numpy.loadtxt(fname='/content/gdrive/My Drive/content/TestData.csv', dtype='str', delimiter=',',
                             encoding='utf-8-sig')
        X = data[:, :-1]  # features are all values but the last on the line
        y = data[:, -1]  # class is the last value on the line
        return X, y


def InorderPrint(root):
    if (root == None):
        return
    print(root.name)
    InorderPrint(root.left)
    InorderPrint(root.right)


def display_information(root):
    print("Decision Tree Formation complete.\n\n:")
    InorderPrint(root)
    print("*** Tree Information ***:")
    print(root.left.name)
    print(root.right.name)
    print(root.left.left.name)
    print(root.left.right.name)
    print(root.left.left.left.name)
    print(root.left.left.right.name)


def InformationGain(entropyAttributes, gainAttributes):
    entropy = EntropyFormula(entropyAttributes)[0]
    entropy_total = EntropyFormula(entropyAttributes)[3]
    for attr in gainAttributes:
        curEntropy = EntropyFormula([attr])[0]
        curTotal = EntropyFormula([attr])[3]
        entropy -= (curTotal / entropy_total) * curEntropy
    return entropy


def calc_entropy(metric, var1, var2):
    total = 0
    for x in metric:
        total += metric[x]
        pos = float(metric[var1] / total)
        neg = float(metric[var2] / total)
    return (-(pos) * math.log2(pos)) - neg * math.log2(neg)


def compare_entropy(o_entropy, feature):
    pass


def get_metrics(y):
    metrics = {}
    for i in y:
        if i not in metrics:
            metrics[i] = 1
        else:
            metrics[i] += 1
    return metrics


prediction = ["RED", "DOMESTIC", "SUV", "==", "NO"]


def DecisionTreeInfo(node, left, right):
    print("Root INIT")
    print(str(node.name))
    print(str(node.branches))
    print(str(node.features))
    print(str(node.entropy))
    if (node.name != "root"):
        print(str(node.parent.name))
    else:
        print(True)
        print(str(node.left.name))
        print(str(node.right.name))
        print(str(node.stolen))
        print(str(left.name))
        print(str(left.branches))
        print(str(left.features))
        print(str(left.entropy))
        print(str(left.parent.name))
        if (left.left == None):
            print(str(left.left))
        else:
            print(str(left.left.name))
            if (right.right == None):
                print(str(right.right))
            else:
                print(str(right.right.name))
                print(str(left.stolen))
                print(str(right.name))
                print(str(right.branches))
                print(str(right.features))
                print(str(right.entropy))
                print(str(right.parent.name))
                print(str(right.left))
                print(str(right.right))
                print(str(right.stolen))
                print("\n\n")


def oldmain():
    X, y = load_data('train')
    ID = X[:, 0]
    color = X[:, 1]
    typecar = X[:, 2]
    origin = X[:, 3]
    color_met = get_metrics(color)
    type_met = get_metrics(typecar)
    origin_met = get_metrics(origin)
    stolen_met = get_metrics(y)
    prior_entropy(color_met, 'Color')
    prior_entropy(type_met, 'Type')
    prior_entropy(origin_met, 'Origin')
    prior_entropy(stolen_met, 'Stolen')
    print(color_met)
    print(type_met)
    print(origin_met)
    print(stolen_met)
    entropy_stolen = (EntropyFormula(stolen_met, "Yes", "No"))
    entropy_color = EntropyFormula(color_met, "Red", "Yellow")
    entropy_type = EntropyFormula(type_met, "Sports", "SUV")
    entropy_origin = EntropyFormula(origin_met, "Domestic", "Imported")


def prior_entropy(mydict, k):
    try:
        del mydict[k]
    except KeyError:
        print("DNE")


def drawTree():
    msg = ""
    print("\t\tTYPE\t\t")
    print("\t/\n\tOrigin\tOrigin(-)\n\t/ \ \nColor(+, -)\tColor(-,+)")
    for i in prediction:
        msg += i
        msg += " "
    print("\n")
    print(msg)


def initTree(node, root):
    info_gain = {}
    print(str(root.features))
    print(str(node.features))
    if (node.features == []):
        print(root.features)
        node.default()
        print(root.features)
        return
    node.set_entropy(EntropyFormula(node.branches)[0])
    for ft in node.features:
        if (ft == "color"):
            info_gain[ft] = InformationGain(node.branches, color_metrics)
        elif (ft == "type"):
            info_gain[ft] = InformationGain(node.branches, type_metrics)
        else:
            info_gain[ft] = InformationGain(node.branches, origin_metrics)
    gainMaxAttr = max(info_gain.items(), key=operator.itemgetter(1))[0]
    node.cleanBranch(gainMaxAttr)
    if (gainMaxAttr == "color"):
        if (node.left == None):
            node.left = Node(color_metrics[0], node.features, node)
        else:
            initTree(node.left, root)

        if (node.right == None):
            node.right = Node(color_metrics[1], node.features, node)
        else:
            initTree(node.right, root)
    elif (gainMaxAttr == "type"):
        if (node.left == None):
            node.left = Node(type_metrics[0], node.features, node)
        else:
            initTree(node.left, root)
        if (node.right == None):
            node.right = Node(type_metrics[1], node.features, node)
        else:
            initTree(node.right, root)
    else:
        if (node.left == None):
            node.left = Node(origin_metrics[0], node.features, node)
        else:
            initTree(node.left, root)
        if (node.right == None):
            node.right = Node(origin_metrics[1], node.features, node)
        else:
            initTree(node.right, root)
    print("Deciding on next feature. . .")
    node.left.buildaBranch(node.left.name)
    node.right.buildaBranch(node.right.name)
    node.left.set_entropy(EntropyFormula(node.left.branches)[0])
    node.right.set_entropy(EntropyFormula(node.right.branches)[0])
    DecisionTreeInfo(node, node.left, node.right)
    initTree(node.left, root)
    initTree(node.right, root)


def printTree(curNode):
    if (curNode == None):
        return
    print(curNode.name)
    printTree(curNode.left)
    printTree(curNode.right)


def calc_entropy(metric, var1, var2):
    total = 0
    for x in metric:
        total += metric[x]
    pos = float(metric[var1] / total)
    neg = float(metric[var2] / total)
    return (-(pos) * math.log2(pos)) - neg * math.log2(neg)


def main():
    root = Node("root")
    print(True)
    initTree(root, root)
    display_information(root)
    drawTree()


if __name__ == '__main__':
    main()
