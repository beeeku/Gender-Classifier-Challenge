from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier

dtc = tree.DecisionTreeClassifier()
knc = KNeighborsClassifier(n_neighbors=3)
svmc = svm.SVC()
gpc = GaussianProcessClassifier(warm_start=True)



#[height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']
dtc = dtc.fit(X, Y)
knc = knc.fit(X, Y)
svmc = svmc.fit(X, Y) 
gpc = gpc.fit(X,Y)


def predictionGenerate(sample):
  data =[]
  accuracy = 0
  dtcPrediction = dtc.predict(sample)
  dtcPredictionScore = dtc.score(X,Y, sample_weight=None)
  kncPrediction = knc.predict(sample)
  kncPredictionScore = knc.score(X,Y, sample_weight=None)
  svmcPrediction = svmc.predict(sample)
  svmcPredictionScore = svmc.score(X,Y, sample_weight=None)
  gpcPrediction = gpc.predict(sample)
  gpcPredictionScore = gpc.score(X,Y, sample_weight=None)

  print dtcPredictionScore
  print kncPredictionScore
  print svmcPredictionScore
  print gpcPredictionScore

  if dtcPredictionScore >accuracy:
    accuracy = dtcPredictionScore
    result = dtcPrediction
    algo = 'dtc'

  
  if kncPredictionScore >accuracy:
    accuracy = kncPredictionScore
    result = kncPrediction
    algo = 'knc'
  
  

  if svmcPredictionScore >accuracy:
    accuracy = svmcPredictionScore
    result = svmcPrediction
    algo = 'svmc'
  
 
  if gpcPredictionScore > accuracy:
    accuracy = gpcPredictionScore
    result = gpcPrediction  
    algo ='gpc'
    
  data.append( result )
  data.append(accuracy)
  data.append(algo) 
  return data 
      

print predictionGenerate([[175, 110, 43]])