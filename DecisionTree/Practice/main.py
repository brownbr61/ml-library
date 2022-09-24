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

# create binary decision tree
def binaryTreeBuilder():
  treeName = "binary"
  dataDirName = os.path.join(dataDirectory, treeName)
  logDirName = os.path.join(logDirectory, treeName)
  try:
    shutil.rmtree(logDirName)
    if os.path.isdir(logDirName) == False:
      os.mkdir(logDirName)
  except:
    print('Error deleting:  {}'.format(logDirName))
  logFileName = "main"
  Log = LumberJack(logDirName,logFileName)
  Log.header("Welcome to binaryTreeBuilder()!")

  Log.comment("Extracting labels from: {}".format(dataDirName + labelFile))
  aliAss = AliAsses(dataDirName)
  binDict = aliAss.aliasDict
  Log.dictionary("binDict",binDict)

  Log.comment("Using binDict to extract labels from {}".format(dataDirName + trainingFile))
  dataMatrix = aliAss.trainDataMatrix
  Log.matrix("dataMatrix",dataMatrix)

  Log.comment("Using binDict to return to labels from dataMatrix")
  dataTable = aliAss.trainDataTable
  Log.matrix("dataTable",dataTable)

  Log.comment("Building Tree:")
  builder = TreeBuilder(logDirName, aliAss, dataMatrix, "root")
  binaryDecisionTree = builder.build()
  Log.tree("Binary Tree:", binaryDecisionTree)

  print("Binary Decision Tree for {}".format(dataDirName + "train.csv"))
  print(RenderTree(binaryDecisionTree, style=AsciiStyle()).by_attr())



def hw1_1a():
  treeName = "1a"
  dataDirName = os.path.join(dataDirectory, treeName)
  logDirName = os.path.join(logDirectory, treeName)
  try:
    shutil.rmtree(logDirName)
  except:
    print('Error deleting:  {}'.format(logDirName))
  if os.path.isdir(logDirName) == False:
    os.mkdir(logDirName)
  logFileName = "main"
  Log = LumberJack(logDirName,logFileName)
  Log.header("Welcome to binaryTreeBuilder()!")

  Log.comment("Extracting labels from: {}".format(dataDirName + labelFile))
  aliAss = AliAsses(dataDirName)
  binDict = aliAss.aliasDict
  Log.dictionary("binDict",binDict)

  Log.comment("Using binDict to extract labels from {}".format(dataDirName + trainingFile))
  dataMatrix = aliAss.trainDataMatrix
  Log.matrix("dataMatrix",dataMatrix)

  Log.comment("Using binDict to return to labels from dataMatrix")
  dataTable = aliAss.trainDataTable
  Log.matrix("dataTable",dataTable)

  Log.comment("Building Tree:")
  builder = TreeBuilder(logDirName, aliAss, dataMatrix, "root")
  binaryDecisionTree = builder.build()
  Log.tree("Binary Tree:", binaryDecisionTree)



binaryTreeBuilder()
# hw1_1a()
