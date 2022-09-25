import numpy
import os
from anytree import RenderTree,AsciiStyle
 # get it? He logs!!!!

class LumberJack:
  def __init__(self,filePath,fileName=""):
    self.ext = ".log"
    self.dirName = filePath
    self.fileName = os.path.join(filePath,fileName + self.ext)
    print("Logging data to:\t" + self.fileName)
    self.fid = open(self.fileName, "w")

  
  def header(self, content):
    self.fid.write(content)
    self.fid.write('\n')

  def comment(self, content):
    self.fid.write("  " + content)
    self.fid.write('\n')

  def matrix(self, label, matrix, tab = 1):
    self.fid.write(self.tabbing(tab))
    self.fid.write(label + " :\n")
    matStr = str(matrix).split('\n')
    matStr = self.tabbing(tab+1) + ('\n' + self.tabbing(tab+1)).join(matStr)
    self.fid.write(matStr)
    self.fid.write('\n')

  def array(self, label, array, tab = 1):
    self.fid.write(self.tabbing(tab))
    self.fid.write(label + " :  ")
    for row in array:
      self.fid.write(self.tabbing(tab+1))
      self.fid.write(str(row) + ';\n')
  
  def dictionary(self, label, dictionary):
    self.fid.write(self.tabbing(1))
    self.fid.write(label + " :  ")
    if(len(dictionary) > 1):
      self.fid.write('\n')
    for key in dictionary:
      if(type(dictionary[key] == numpy.ndarray)):
        self.matrix(str(key),dictionary[key],2)
      else:
        self.fid.write(self.tabbing(2))
        self.fid.write(str(key) + " :\t")
        self.fid.write(str(dictionary[key]))
        self.fid.write('\n')

  def tabbing(self,tab = 1):
    myString = ""
    for i in range(0,tab):
      myString = myString + "  "
    return myString
  
  def tree(self, label, root):
    self.fid.write(label + " :\n")
    self.fid.write(RenderTree(root, style=AsciiStyle()).by_attr())
    self.fid.write("\n")