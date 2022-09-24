from ast import alias
import numpy
import os
import matplotlib as mpl
import pandas as panda

class AliAsses:
  def __init__(self, dir):
    self.aliasDict = self.getLabels(os.path.join(dir,"labels.txt"))
    self.trainDataMatrix = self.rmLabels(os.path.join(dir,"train.csv"))
    self.trainDataTable = self.relabelData(self.trainDataMatrix)
  
  def keys(self):
    return list(self.aliasDict.keys())

  def labelKey(self):
    return list(self.aliasDict.keys())[-1]

  def getLabels(self, filename) -> dict:
    aliasDict = dict()
    with open(filename, 'r') as f:
      for line in f:
        terms = line.strip().split(',')
        # key, value0, value1
        for i in range(0,len(terms)):
          terms[i] = terms[i].strip()
        aliasDict[terms[0]] = list(terms[1:])
    return aliasDict

  def rmLabels(self, filename) -> numpy.ndarray:
    dataMatrix = []
    with open(filename,'r') as f:
      for line in f:
        matRow = []
        terms = line.strip().split(',')
        for i in range(0,len(terms)):
          # turn Dictionary Keys into addressable list
          columnKey = list(self.aliasDict.keys())[i]
          # create matrixRow to later append onto dataMatrix
          matRow.append(self.aliasDict[columnKey].index(terms[i].strip()))
        dataMatrix.append(matRow)
    return numpy.array(dataMatrix)

  def relabelData(self, dataMatrix) -> list:
    dataTable = []
    for row in dataMatrix:
      datRow = []
      for i in range(0,len(self.aliasDict.keys())):
        # turn Dictionary Keys into addressable list
        columnKey = list(self.aliasDict.keys())[i]
        # create dataRow to later append onto dataMatrix
        datRow.append(self.aliasDict[columnKey][row[i]])
      dataTable.append(datRow)
    return dataTable
    




