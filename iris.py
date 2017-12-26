# problem: predict class of iris plant given lengths and widths of sepal and petal

# Attribute Information:
#   1. sepal length in cm
#   2. sepal width in cm
#   3. petal length in cm
#   4. petal width in cm
#   5. class: 
#      -- Iris Setosa
#      -- Iris Versicolour
#      -- Iris Virginica

# Check the versions of libraries
print("= 1.2 Start Python and Check Versions =")

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# Load libraries
print(" = 2. Load The Data = ")
print(" == 2.1 Load libraries == ")
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
print(" == 2.2 Load dataset == ")
url  =  "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names  =  ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset  =  pandas.read_csv(url, names = names)

print(" = 3. Summarize the Dataset = ")
# shape
print(" == 3.1 Dimensions of Dataset, shape of data == ")
print(dataset.shape)

# head
print(" == 3.2 Peek at the Data, head -- first 10 items == ")
print(dataset.head(10))

# descriptions
print(" == 3.3 Statistical Summary == ")
print(dataset.describe())

# class distribution
print(" == 3.4 class distribution  ==")
print(dataset.groupby('class').size())

print(" = 4. Data Visualization = ")
# box and whisker plots
print(" == 4.1 Univariate Plots: box and whisker plots. why? to determine outliers? = ")
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
print(" == 4.1 Univariate Plots: histograms. why? to determine if the distribution is normal-like? == ")
dataset.hist()
plt.show()

# scatter plot matrix
print("== 4.2 Multivariate Plots: Multivariate Plots:scatter plot matrix. why? to spot structured relationships between input variables ==")
scatter_matrix(dataset)
plt.show()

print(" = 5. Evaluate Some Algorithms = ")
# Split-out validation dataset
print(" == 5.1 Create a Validation Dataset: Split-out validation dataset == ")
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
print(" == 5.2 Test Harness: Test options and evaluation metric == ")
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
print("== 5.3 Build Models: build and evaluate our five models, Spot Check Algorithms ==")
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
print("algorithm",scoring,"mean","std")
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	
	
# Compare Algorithms
print(" == 5.4 Select Best Model, Compare Algorithms == ")
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111) # what does 111 mean?
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Make predictions on validation dataset
print("= 6. Make Predictions: Make predictions on validation dataset =")
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
