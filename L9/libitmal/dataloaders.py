#!/usr/bin/env python3

import os
import matplotlib
from matplotlib import pyplot as matplotlib_pyplot
from sklearn import datasets as sklearn_datasets

def IRIS_GetDataSet():
	iris = sklearn_datasets.load_iris()
	X = iris.data  # we only take the first two features.
	y = iris.target
	return X, y

def IRIS_PlotFeatures(X, y, i, j):
	matplotlib_pyplot.figure(figsize=(8, 6))
	matplotlib_pyplot.clf()
	matplotlib_pyplot.scatter(X[:, i], X[:, j], c=y, cmap=matplotlib_pyplot.cm.Set1, edgecolor='k')
	matplotlib_pyplot.show()

def MOON_GetDataSet(n_samples=100, noise=0.1, random_state=0):
	X, y=sklearn_datasets.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
	return X, y

def MOON_Plot(X, y):
	figure = matplotlib_pyplot.figure(figsize=(12, 9))
	matplotlib_pyplot.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
	matplotlib_pyplot.show()
	return figure

def MNIST_PlotDigit(data):
	image = data.reshape(28, 28)
	matplotlib_pyplot.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")
	matplotlib_pyplot.axis("off")

## CEF: A production-code ready version of get MNIST
##      Can run in four three different modes, 0=auto, 1=tensorflow.keras, 2=keras and 3=fetch_openml (slow and disfavored)
def MNIST_GetDataSet(reshape784=True, load_mode=0):
	# NOTE: notice that X and y are defined inside if's, not in outer scope as in C++, strange
	# NOTE: hardcoded sizes, 70000 x 28 x 28 or 70000 x 784

	def AssertShapesAndTypesOk(test_reshape784=True):
		assert 2 <= X.ndim <= 3
		assert X.shape[0]==70000
		assert X.shape[0]==y.shape[0]
		assert (X.ndim==2 and X.shape[1]==784) or (X.ndim==3 and X.shape[1]==28 and X.shape[2]==28)
		assert not test_reshape784 or (reshape784 and X.ndim==2) or (not reshape784 and X.ndim==3)
		assert y.ndim==1
		assert X.dtype=='uint8' or (X.dtype=='float64' and load_mode==-2)
		assert y.dtype=='uint8'

	import numpy as np
	import warnings

	if  load_mode<0 or load_mode>3:
		raise ValueError("load_mode must be 0 (auto), 1 (keras), 2(tensorflow.keras), or 3 (fetch_openml)")

	if load_mode in [0, 1]:
		try:
			from tensorflow.keras.datasets import mnist as tensorflow_keras_mnist
			(X_train, y_train), (X_test, y_test) = tensorflow_keras_mnist.load_data()
			load_mode=-1
		except Exception as ex:
			if load_mode==0:
				warnings.warn("MNIST_GetDataSet(): failed to import and load data in load_mode 'tensorflow.keras', proceding to next mode..")
			else:
				raise ImportError("Can not run in tensorflow.keras mode due to missing tensorflow installation") from ex

	if load_mode in [0, 2]:
		try:
			from keras.datasets import mnist as keras_mnist
			(X_train, y_train), (X_test, y_test) = keras_mnist.load_data()
			load_mode=-1
		except Exception as ex:
			if load_mode==0:
				warnings.warn("MNIST_GetDataSet(): failed to import and load data in load_mode 'keras', proceding to next mode..")
			else:
				raise ImportError("Can not run in keras mode due to missing keras installation, you might want to try load_mode=1 for tensorflow.keras mode instead") from ex

	if load_mode in [0, 3]:
		try:
			from sklearn.datasets import fetch_openml
			d = fetch_openml('mnist_784', version=1, cache=True, as_frame=False) # Note: default as_frame changed from False to 'auto' in 0.24.
			X, y= d["data"], d["target"]

			warnings.warn("MNIST_GetDataSet(): fetch openml mode is slow and uses 'float64' instead of 'uint8'")

			if y.dtype!='uint8':
				assert y.shape[0]>0
				warnings.warn(f"MNIST_GetDataSet(): fetch openml mode converts y from '{type(y[0])}' to 'uint8'")

			y = y.astype('uint8')
			load_mode=-2
		except Exception as ex:
			raise ImportError("Can not fun in fetch ml mode, due to failed call to fech_openml(..)") from ex

	if load_mode==-1:
		X, y = np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test))
	elif load_mode==-2:
		pass
	else:
		raise ImportError("You do not have Keras or Tensorflow.Keras installed, so keras.datasets.mnist.load_data() or tensorflow.keras.datasets.mnist.load_data() will not work!")

	AssertShapesAndTypesOk(False)

	if not reshape784 and X.ndim==2:
		assert X.shape[1]==784
		X=np.reshape(X, (70000, 28, 28))
		assert X.ndim==3
		assert X.shape[1]==28 and X.shape[2]==28

	if reshape784 and X.ndim==3:
		assert X.shape[1]==28 and X.shape[2]==28
		X=np.reshape(X, (70000, 784))
		assert X.ndim==2
		assert X.shape[1]==784

	AssertShapesAndTypesOk()

	return X, y

# Final fallback method, when everything else fails:
#   load from a zipped pickle file, that is stored locally, you need to have
#   the MNIST_data.pkl.gz pre-generetated via the GetMNISTDataAndStoreAsPickle()
#   above.

def MNIST_GetDataSet_via_pickle_file(filename = "MNIST_data.pkl.gz"):
	if not os.path.exists(filename):
		raise FileNotFoundError(f"ERROR: file {filename}' does not exists or is not readable")

	try:
		import pickle
		import gzip

		print(f"Loading MNIST data from filename '{filename}'..")

		with gzip.open(filename, "rb") as f:
			X, y = pickle.load(f)

			print(f"  X.dtype={X.dtype}")
			print(f"  X.shape={X.shape}")
			print(f"  y.shape={y.shape}")

			print("OK")
			return X, y
	except Exception as ex:
		raise RuntimeError(f"ERROR: could not run MNIST_GetDataSet_via_pickle_file() due to exception {ex}..") from ex

######################################################################################################
#
# TESTS
#
######################################################################################################

def Test_IRIS_GetDataSet(debug=False, verbose=False):
	if debug:
		print("Test_IRIS_GetDataSet()..")
	X, y = IRIS_GetDataSet()
	if verbose:
		IRIS_PlotFeatures(X, y, 0, 1)

def Test_MOON_GetDataSet(debug=False, verbose=False):
	if debug:
		print("Test_MOON_GetDataSet()..")
	X, y = MOON_GetDataSet()
	if verbose:
		MOON_Plot(X, y)

def Test_MNIST_GetDataSet(debug=False, verbose=False):
	if debug:
		print("Test_MNIST_GetDataSet()..")
	# Local function, pretty neat huh?
	def PrintShapeAndType(X, y, n):
		print('')
		print(f'X{n}.shape={X.shape}), X{n}.dtype={X.dtype}')
		print(f'y{n}.shape={y.shape}), y{n}.dtype={y.dtype})')

		assert X.shape[0]==y.shape[0]
		assert 2 <= X.ndim <= 3
		assert (X.ndim==2 and X.shape[1]==784) or (X.ndim==3 and X.shape[1]==28 and X.shape[2]==28)

	def AssertEqual(Z1, Z2, check_dtypes=True):
		assert Z1.ndim==Z2.ndim, f'unequal dimensions Z1.ndim={Z1.ndim}, Z2.ndim={Z2.ndim}'
		assert type(Z1)==type(Z2), f'diff types, type(Z1)={type(Z1)}, type(X2)={type(Z2)}'
		if check_dtypes:
			assert Z1.dtype==Z2.dtype, f'diff dtypes, Z1.dtype={Z1.dtype}, Z2.dtype={Z2.dtype}'

		for i in range(0, Z1.ndim):
			assert Z1.shape[i]==Z2.shape[i], f'unequal shapes, Z1.shape[{i}]={Z1.shape[i]}, Z2.shape[{i}]={Z2.shape[i]}'

	X1, y1=MNIST_GetDataSet(reshape784=True,  load_mode=0)
	X2, y2=MNIST_GetDataSet(reshape784=False, load_mode=0)
	X3, y3=MNIST_GetDataSet(reshape784=True,  load_mode=1)
	X4, y4=MNIST_GetDataSet(reshape784=False, load_mode=1)
	X5, y5=MNIST_GetDataSet(reshape784=True,  load_mode=2)
	X6, y6=MNIST_GetDataSet(reshape784=False, load_mode=2)
	if verbose:
		X7, y7=MNIST_GetDataSet(reshape784=True,  load_mode=3)
		X8, y8=MNIST_GetDataSet(reshape784=False, load_mode=3)

	if debug>0:
		PrintShapeAndType(X1, y1, 1)
		PrintShapeAndType(X2, y2, 2)
		PrintShapeAndType(X3, y3, 3)
		PrintShapeAndType(X4, y4, 4)
		PrintShapeAndType(X5, y5, 5)
		PrintShapeAndType(X6, y6, 6)
		if verbose:
			PrintShapeAndType(X7, y7, 7)
			PrintShapeAndType(X8, y8, 8)

	AssertEqual(X1, X3)
	AssertEqual(X2, X4)
	AssertEqual(X1, X5)
	AssertEqual(X2, X6)
	if verbose:
		AssertEqual(X1, X7, False)
		AssertEqual(X2, X8, False)

	AssertEqual(y1, y2)
	AssertEqual(y1, y3)
	AssertEqual(y1, y4)
	AssertEqual(y1, y5)
	AssertEqual(y1, y6)
	if verbose:
		AssertEqual(y1, y7)
		AssertEqual(y1, y8)

	#assert np.array_equal(X1,X3)
	#assert np.array_equal(X2,X4)
	#assert np.array_equal(y2,y4)
	#assert (X1.ravel()==X2.ravel()).all()

	if verbose:
		MNIST_PlotDigit(X1[1])


def TestAll(debug=False, verbose=False):

	# local function, should not be exposed public:
	def GetMNISTDataAndStoreAsPickle(filename = "MNIST_data.pkl.gz"):
		import pickle
		import gzip

		X, y = MNIST_GetDataSet(reshape784=False)

		print(f"Saving MNIST data to filename '{filename}'..")
		print(f"  X.dtype={X.dtype}")
		print(f"  X.shape={X.shape}")
		print(f"  y.shape={y.shape}")

		with gzip.open(filename, "wb") as f:
			pickle.dump((X, y), f)

		print("OK")

	Test_IRIS_GetDataSet (debug=debug, verbose=verbose)
	Test_MOON_GetDataSet (debug=debug, verbose=verbose)
	Test_MNIST_GetDataSet(debug=debug, verbose=verbose)

	if False: # enable only if you need a local pickle MNIST data file..
		GetMNISTDataAndStoreAsPickle()
		MNIST_GetDataSet_via_pickle_file()

		failed = False
		try:
			MNIST_GetDataSet_via_pickle_file("no_such_file")
		except FileNotFoundError:
			failed = True
		assert failed

	print("ALL OK")

if __name__ == '__main__':
	TestAll()
