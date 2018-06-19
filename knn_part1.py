import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
print(iris.data.shape)
print(iris.target.shape)
