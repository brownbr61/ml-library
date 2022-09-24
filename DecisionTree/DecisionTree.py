from anytree import Node, RenderTree, AsciiStyle

class DecisionTree:
  def __init__(self):
    self.Tree = dict()
    tmp = Node(0)
    Node(1, parent=tmp)
    Node(0, parent=tmp)
    self.root = tmp

  def print(self):
    print(RenderTree(self.root, style=AsciiStyle()).by_attr())
    print(type(self.root))
    