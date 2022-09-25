import os
import numpy
import shutil
import matplotlib
from anytree import RenderTree,AsciiStyle

print("Welcome to hw1 main.py!\n")
print("Working Directory:\t" + os.getcwd())

# Confirm that the root folder is in our python path
import sys
sys.path.insert(0,os.getcwd())
assert (os.getcwd() in sys.path)

from DecisionTree.AliasManager import AliAsses
from MiscTools.LumberJack import LumberJack
from DecisionTree.TreeBuilder import TreeBuilder

trainingFile = "train.csv"
testingFile = "test.csv"
labelFile = "labels.txt"
dataDirectory = "./DecisionTree/Practice/data/"
logDirectory = "./DecisionTree/Practice/logs/"

def cleanDir(dirName):
  try:
    shutil.rmtree(dirName)
    if os.path.isdir(dirName) == False:
      os.mkdir(dirName)
  except:
    print('Error deleting:  {}'.format(dirName))
  if os.path.isdir(dirName) == False:
    os.mkdir(dirName)



def myTreeBuilder(treeName):  
  myDataDir = os.path.join(dataDirectory, treeName)
  myLogDir = os.path.join(logDirectory, treeName)
  cleanDir(myLogDir)
  Log = LumberJack(myLogDir,"main")
  Log.header("This function will build: {}()!".format(treeName))

  Log.comment("Extracting labels from: {}".format(myDataDir + labelFile))
  aliAss = AliAsses(myDataDir)
  aliasDict = aliAss.aliasDict
  Log.dictionary("aliasDict",aliasDict)

  Log.comment("Using aliasDict to extract labels from {}".format(myDataDir + trainingFile))
  dataMatrix = aliAss.trainDataMatrix
  Log.matrix("dataMatrix",dataMatrix)

  Log.comment("Using aliasDict to return to labels from dataMatrix")
  dataTable = aliAss.trainDataTable
  Log.matrix("dataTable",dataTable)

  Log.comment("Building Tree:")
  builder = TreeBuilder(Log.dirName, aliAss, dataMatrix, "root")
  Log.matrix("Training Data:",dataMatrix)
  decisionTree = builder.build()
  print("\nTraining Data: ")
  print(dataMatrix)
  
  treeHeader = "Decision Tree for {}".format(treeName)
  Log.tree(treeHeader, decisionTree)
  print(treeHeader)
  print(RenderTree(decisionTree, style=AsciiStyle()).by_attr())



# create binary decision tree
def binaryTreeBuilder():
  myTreeBuilder("binary")

# create binary decision tree for 1a
def hw1_1a():
  myTreeBuilder("1a")

# create decision tree for 1a
def tennis():
  myTreeBuilder("tennis")

# create decision tree for car
def ternary():
  myTreeBuilder("ternary")

# create decision tree for car
def car():
  myTreeBuilder("car")



binaryTreeBuilder()
hw1_1a()
tennis()
ternary()
# car()
