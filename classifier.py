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


def prediction_generate(sample):
  data =[]
  accuracy = 0
  dtc_prediction = dtc.predict(sample)
  dtc_prediction_score = dtc.score(X,Y, sample_weight=None)
  knc_prediction = knc.predict(sample)
  knc_prediction_score = knc.score(X,Y, sample_weight=None)
  svmc_prediction = svmc.predict(sample)
  svmc_prediction_score = svmc.score(X,Y, sample_weight=None)
  gpc_prediction = gpc.predict(sample)
  gpc_prediction_score = gpc.score(X,Y, sample_weight=None)

  #print dtc_prediction_score
  #print knc_prediction_score
  #print svmc_prediction_score
  #print gpc_prediction_score

  if dtc_prediction_score >accuracy:
    accuracy = dtc_prediction_score
    result = dtc_prediction
    algo = 'dtc'

  
  if knc_prediction_score >accuracy:
    accuracy = knc_prediction_score
    result = knc_prediction
    algo = 'knc'
  
  

  if svmc_prediction_score >accuracy:
    accuracy = svmc_prediction_score
    result = svmc_prediction
    algo = 'svmc'
  
 
  if gpc_prediction_score > accuracy:
    accuracy = gpc_prediction_score
    result = gpc_prediction  
    algo ='gpc'
    
  data.append( result )
  data.append(accuracy)
  data.append(algo) 
  return data 
      

print prediction_generate([[175, 110, 43]])
