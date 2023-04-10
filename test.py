from main import outputGauge
test = ''

outputGauge(test)

for i in range(1,10):
  outputGauge(i)

class NewClass():
  def __init__(self, tmp):
    self.tmp = tmp

  def get(self):
    return self.tmp

class Testclass():

  def __init__(self, var):
    self.var = var

  def setVar(self, newvar):
    self.var = newvar

obj = NewClass(100)
a = obj.get
  
