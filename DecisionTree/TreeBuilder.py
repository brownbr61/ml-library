from copy import deepcopy
import os
from anytree import Node, RenderTree, AsciiStyle
from MiscTools.LumberJack import LumberJack
import numpy

class TreeBuilder:
  def __init__(self, logDir, aliAss, dataMatrix, filename = "node"):
    self.dir = logDir
    if os.path.isdir(self.dir) == False:
      os.mkdir(self.dir)
    self.Log  = LumberJack(logDir, filename)
    self.Log.header("Just in! We've received the following data set!".format(filename))
    self.Log.matrix("data",dataMatrix)

    assert(type(dataMatrix) == numpy.ndarray)
    self.data = dataMatrix
    self.alias = aliAss
    self.Log.dictionary("We've got the following dictionary", self.alias.aliasDict)

    if self.data.size > 0:
      self.labelData = self.data[:,-1]
    else:
      self.labelData = 0
    self.isLeaf = self.isLeafNode()



  def build(self) -> Node:
    p = dict()
    h = dict()

    self.Log.header("Calculate P")
    for attribute in self.alias.keys():
      self.Log.comment("Calculate P(label|{})".format(attribute))
      p[attribute] = self.P(self.alias.labelKey(),attribute)
      self.Log.matrix("p({})".format(attribute), p[attribute])

    self.Log.header("Calculate H")
    for attribute in self.alias.keys():
      self.Log.comment("Calculate H(label|{})".format(attribute))
      h[attribute] = self.H(p[attribute])
      self.Log.comment("h({0}) :  {1}".format(attribute, h[attribute]))
    h[self.alias.labelKey()] = sum(h[self.alias.labelKey()])

    self.Log.header("Calculate I")
    i = []
    for attribute in self.alias.keys()[0:-1]:
      self.Log.comment("Calculate I(label|{})".format(attribute))
      iTmp = self.I(p[attribute], h[attribute])
      i.append(iTmp)
      self.Log.comment("i({0}) :  {1}".format(attribute, iTmp))
    
    self.Log.header("Determine max(iGain)")
    i = numpy.array(i)
    iGain = h[self.alias.labelKey()] - i
    maxI = numpy.argmax(i)
    self.Log.comment("iGain :  {}".format(iGain))
    self.Log.comment("max(iGain) :  {}".format(maxI))

    self.Log.header("Split Data sets by column of max(iGain):")
    key2sort = self.alias.keys()[maxI]
    self.Log.comment("label of max(iGain) :  {0}->{1}".format(key2sort, \
      self.alias.aliasDict[key2sort]))
    self.Log.comment("label of max(iGain) :  {}".format(key2sort))
    self.Log.comment("dict keys: {}".format(self.alias.keys()))
    splitDataSets = dict()
    for value in range(0,len(self.alias.aliasDict[key2sort])):
      splitDataSets[value] = self.splitData(maxI,value)
    self.Log.dictionary("split data sets:",splitDataSets)
    self.Log.comment("Now we eliminate column of iGain")
    for key in splitDataSets:
      if splitDataSets[key].size > 0:
        splitDataSets[key] = numpy.delete(splitDataSets[key],maxI,1)
    alias = deepcopy(self.alias)
    del alias.aliasDict[key2sort]
    self.Log.dictionary("alias dictionary",alias.aliasDict)
    self.Log.dictionary("split data sets:",splitDataSets)

    self.Log.header("Create & Call a TreeBuilder for each new data set. Then build trees")
    treeBuilders = []
    orphanTrees = []
    for dataSet in splitDataSets:
      self.Log.comment("Create new dir for new nodes:")
      newDir = os.path.join(self.dir,"{}.{}".format(key2sort,str(self.alias.aliasDict[key2sort][dataSet])))
      builder = TreeBuilder(newDir,alias,splitDataSets[dataSet])
      treeBuilders.append(builder)
      if builder.isLeaf:
        self.Log.comment("Found a leaf!")
        node = builder.createLeaf()
      else:
        self.Log.comment("calling builder.build()")
        node = builder.build()
      orphanTrees.append(Node(self.alias.aliasDict[key2sort][dataSet], children=[node]))
      self.Log.header(RenderTree(orphanTrees[-1], style=AsciiStyle()).by_attr())
      self.Log.dictionary("alias dictionary",alias.aliasDict)
    
    self.Log.header("Giving orphan trees a home!")
    root = Node(key2sort, children=orphanTrees)
    self.Log.header(RenderTree(root, style=AsciiStyle()).by_attr())

    self.Log.header("Closing build()")
    print("Done logging data to:\t {}".format(self.dir))
    return root



  def createLeaf(self):
    self.Log.header("Creating Leaf Node:")
    root = Node(-1)
    attributes = self.alias.aliasDict[self.alias.keys()[0]]
    labelist = self.alias.aliasDict[self.alias.keys()[-1]]
    self.Log.matrix("labelist",labelist)

    if self.data.size == 0:
      self.Log.header("NO Data here:")

    if self.allLabelsEqual():
      self.Log.header("All Labels are identical:")
      self.Log.matrix("labels",self.labelData)
      leafVal = Node(labelist[self.labelData[0]])
      root = Node(self.alias.keys()[-1], children=[leafVal])

    elif self.data.shape[0] == 0: # there are no rows of data left
      self.Log.header("No data remaining; oops!:")
      self.Log.matrix("labels",self.labelData)
      root = Node(self.alias.keys()[-1], children=[Node(labelist[0])])

    elif self.data.shape[0] == 1: # there's only one row of data left
      self.Log.header("Only row of data Remaining:")
      self.Log.matrix("labels",self.labelData)
      root = Node(labelist[self.labelData[0]])
      root = Node(self.alias.keys()[-1], children=[leafVal])

    elif self.data.shape[1] == 2:
      self.Log.header("Only one column of attributes remaining:")
      self.Log.dictionary("aliases", self.alias.aliasDict)
      self.Log.matrix("data", self.data)
      # get the probability matrix and use it to assign values to child nodes
      p = self.P(self.alias.keys()[0],self.alias.keys()[-1])
      nodes = []
      for i in range(0, len(self.alias.aliasDict[self.alias.keys()[0]])):
        # assign max probability label assignment as child node value
        if i < p.shape[0]:
          leafVal = Node(labelist[numpy.argmax(p[i,:])])
        else:
          leafVal = Node(labelist[numpy.argmax(numpy.sum(p, axis=0))])
        labelNode = Node(self.alias.keys()[-1], children=[leafVal])
        nodes.append(Node(attributes[i], children=[labelNode]))
      root = Node(self.alias.keys()[0],children=nodes)

    self.Log.tree("Returning Leaf node:",root)
    self.Log.header("Closing out of createLeaf()")
    print("Done logging data to:\t {}".format(self.dir))
    return root



  def splitData(self,column,value):
    splitDataSet = []
    for row in self.data:
      if row[column] == value:
        splitDataSet.append(row)
    return numpy.array(splitDataSet)



  def H(self, p):
    p = (p == 0)*1 + p
    return -1*numpy.sum(numpy.multiply(p,numpy.log(p)),1)


  
  def I(self, p, h):
    return sum(numpy.matmul(h.transpose(),p))

  # calculates P(Y|X)
  def P(self, Y, X) -> numpy.array:
    if Y == X:
      return(self.pSet(X))
    else:
      return(self.pCond(Y,X))



  def pSet(self, X):
    xLabelCt = len(self.alias.aliasDict[X])
    xCol = self.alias.keys().index(X)
    p = numpy.zeros([xLabelCt,1])
    for x in range(0,xLabelCt):
      p[x] = numpy.sum((self.data[:,xCol] == x)*1)/len(self.data[:,xCol])
    return(p)


  
  def pCond(self, Y, X):
    xLabelCt = len(self.alias.aliasDict[X])
    yLabelCt = len(self.alias.aliasDict[Y])

    xCol = self.alias.keys().index(X)
    yCol = self.alias.keys().index(Y)

    p = numpy.zeros([xLabelCt,yLabelCt])
    for x in range(0,xLabelCt):
      for y in range(0,yLabelCt):
        xMat  = (self.data[:,xCol] == x) * 1
        yMat  = (self.data[:,yCol] == y) * 1
        sxMat = sum(xMat) 
        sxMat = sxMat + (sxMat + .00001)
        p[x,y] = xMat.dot(yMat)/sxMat
    return p
    


  def isLeafNode(self):
    return( 
            self.data.size == 0 or \
            self.data.shape[0] <= 1 or \
            self.data.shape[1] == 2 or \
            self.allLabelsEqual()
          )



  def allLabelsEqual(self):
    value = self.labelData[0]
    for val in self.labelData:
      if (val == value) == False:
        return False
    return True



  def squirrel(self, node, column):
    for child in node.children:
      if child.children:
        child.name = child.name + (child.name >= column)
        child = self.squirrel(child, column)
    return node