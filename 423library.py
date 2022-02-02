import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

class MappingTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, mapping_column, mapping_dict:dict):  
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'MappingTransformer.transform unknown column {self.mapping_column}'
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
#############################################################################################################################
class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=True):  
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first
  #fill in the rest below
  def fit(self, X, y = None):
    print(f"Warning: {self.__class__.__name__}.fit does nothing.")
    return self

  def transform(self, X):
    X = pd.get_dummies(X,
                               prefix= self.target_column,    #your choice
                               prefix_sep='_',     #your choice
                               columns=[self.target_column],
                               dummy_na= self.dummy_na,    #will try to impute later so leave NaNs in place
                               drop_first= self.drop_first    #really should be True but I wanted to give a clearer picture
                               )
    return X


  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
#############################################################################################################################
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
    assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
    self.column_list = column_list
    self.action = action

  #fill in rest below
  def fit(self, X, y = None):
    print(f"Warning: {self.__class__.__name__}.fit does nothing.")
    return self

  def transform(self, X):
    X_ = X
    if self.action == 'drop':
      X_ = X_.drop(self.column_list, axis=1)
    else:
      X_ = X_.drop(X_.columns.difference(self.column_list), axis=1)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
#############################################################################################################################
class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):  
    self.target_column = target_column

  def fit(self, X, y = None):
    print(f"Warning: {self.__class__.__name__}.fit does nothing.")
    return X

  def transform(self, X):
    X_ = X
    #compute mean of column - look for method
    m = X[self.target_column].mean()
    #compute std of column - look for method
    sigma = X[self.target_column].std()
    minb, maxb = (m-3*sigma, m+3*sigma)  #(lower bound, upper bound)
    X_[self.target_column] = X[self.target_column].clip(lower=minb, upper=maxb)
    return X_
 

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
#############################################################################################################################  
class TukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence='outer'):
    assert fence in ['inner', 'outer']
    self.target_column = target_column
    self.fence = fence

  def fit(self, X, y = None):
    print(f"Warning: {self.__class__.__name__}.fit does nothing.")
    return X

  def transform(self, X):
    X_ = X.copy()
    q1 = X_[self.target_column].quantile(0.25)
    q3 = X_[self.target_column].quantile(0.75)
    iqr = q3-q1
    inner_low = q1-(1.5*iqr)
    inner_high = q3+(1.5*iqr)
    outer_low = q1-3*iqr
    outer_high = q3+3*iqr
    if self.fence == 'inner':
      X_[self.target_column] = X_[self.target_column].clip(lower=inner_low, upper=inner_high)
    elif self.fence == 'outer':
      X_[self.target_column] = X_[self.target_column].clip(lower=outer_low, upper=outer_high)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
############################################################################################################################# 

class MinMaxTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    self.column_min_max = dict()

  #fill in rest below
  def fit(self, X, y = None):
    return X

  def transform(self, X):
    X_ = X
    for i in X_.columns:
      min = X_[i].min()
      max = X_[i].max()
      denom = (max - min)
      new = [(j - min)/denom for j in X_[i]]
      X_[i] = new
    return X_

  
  def fit_transform(self, X, y = None):
    self.fit(X,y)
    result = self.transform(X)
    return result
  
 #############################################################################################################################
class KNNTransformer(BaseEstimator, TransformerMixin):
  def __init__(self,n_neighbors=5, weights="uniform", add_indicator=False):
    self.n_neighbors = n_neighbors
    self.weights=weights 
    self.add_indicator=add_indicator

  #your code
  def fit(self, X, y = None):
    print(f"Warning: {self.__class__.__name__}.fit does nothing.")
    return self

  def transform(self, X):
    X_ = X
    knni = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights, add_indicator=self.add_indicator)
    columns = X_.columns
    matrix = knni.fit_transform(X_)
    result_df = pd.DataFrame(matrix,columns=columns)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  #############################################################################################################################
  
def nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_random_state(df, labels, n=200):
  model = LogisticRegressionCV(random_state=1, max_iter=5000)
  var = []  #collect test_error/train_error where error based on F1 score
  for i in range(1, n):
      train_X, test_X, train_y, test_y = train_test_split(transformed_df, labels, test_size=0.2, shuffle=True,
                                                      random_state=i, stratify=labels)
      model.fit(train_X, train_y)  #train model
      train_pred = model.predict(train_X)  #predict against training set
      test_pred = model.predict(test_X)    #predict against test set
      train_error = f1_score(train_y, train_pred)  #how bad did we do with prediction on training data?
      test_error = f1_score(test_y, test_pred)     #how bad did we do with prediction on test data?
      error_ratio = test_error/train_error        #take the ratio
      var.append(error_ratio)

  rs_value = sum(var)/len(var)
  nearest_val = nearest(var, rs_value)
  return var.index(nearest_val)
