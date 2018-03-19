Flazy
=====

*This software is at a very early stage of development and is not at all complete or stable*

Flazy is a [Spark](http://spark.apache.org/)-inspired Dataset management library for machine learning in Python, providing functional APIs and convenience methods for reading and writing from various data formats. Its intended use cases are:

- Short and convenient code for reading datasets for ML, without a vendor lock-in
- Streaming data into ML pipelines with on-the-fly preprocessing/augmentation
- Randomized sampling from large datasets that does not fit into memory (a la [Pescador](https://github.com/pescadores/pescador))
- ETL-style data munging that fits reasonably in a machine (< 1TB, I'd say)


Basic Usage
-----------

All functionalities of Flazy are accessible through the `Dataset` class:

```python
from flazy import Dataset

Dataset(range(10)).foreach(print)  # prints numbers 0-9
```

If you are familiar with Spark's `RDD` or `tf.data.Dataset`, its functional transformation APIs should look familiar:

```python
Dataset(range(10)).filter(lambda x: x % 2 == 1).foreach(print)  # prints 1, 3, 5, 7, 9
```

`Dataset` itself is a Python `iterable`, which you can use like any collections:

```python
sum(Dataset(range(10)).map(lambda x: x + 1))  # sum of integers up to 10: 55
```

`Dataset` comes with a handful of methods for lazy-evaluated functional transformations. For the full list, please refer to [the source code](flazy/datasets.py) for now, until the documentation site is ready.


Reading Data
------------

If the data fits in the memory, the easiest way to create a `Dataset` is to pass the lists as its arguments. If multiple of them are given, the dataset behaves as a collection of tuples:

```python
import numpy as np

data = Dataset(np.array([[1, 2, 3], [4, 5, 6]]), [0, 1])
data.foreach(print)

# output:
#     (array([1, 2, 3]), 0)
#     (array([4, 5, 6]), 1)
```

When they are fed as keyword arguments, the dataset emits dictionaries instead:

```python
data = Dataset(features=np.array([[1, 2, 3], [4, 5, 6]]), labels=[0, 1])
data.foreach(print)

# output:
#     {'features': array([1, 2, 3]), 'labels': 0}
#     {'features': array([4, 5, 6]), 'labels': 1}
```

Please note that it naturally splits the first dimension of the numpy array, treating it as an iterable over the other dimensions. This is suitable when you have a dataset in a single chunk of Numpy array.

`Dataset` has helper methods to read directly from a supported format, such as `tfrecord`s:

```python
data = Dataset.read.tfrecord('nsynth-test.tfrecord', keys=['audio', 'pitch'], compression='gzip')
print(data.first())

# output:
# {'audio': array([ 3.8138387e-06, -3.8721851e-06,  3.9331076e-06, ...,
#       -3.6526076e-06,  3.7041993e-06, -3.7578957e-06], dtype=float32), 'pitch': array([100], dtype=int64)}
```


Feeding Data
------------

Before feeding, you might want to create a mini-batch of the samples, which is incredibly easy:

```python
data = Dataset(a=range(10), b=range(10, 20))
data.batch(4).foreach(print)

# output:
#     {'a': array([0., 1., 2., 3.], dtype=float32), 'b': array([10., 11., 12., 13.], dtype=float32)}
#     {'a': array([4., 5., 6., 7.], dtype=float32), 'b': array([14., 15., 16., 17.], dtype=float32)}
```

Dataset recognizes whether each item is a dictionary or tuples, and builds a Numpy array with a batch dimension prepended.

If you just need to reconstruct the whole data as a single object after some transformations, there are two methods:

```python
data = Dataset(a=range(10), b=range(10, 20))
data.list()     # returns the list of dicts: [{'a': 0, 'b': 10}, ..., {'a': 9, 'b': 19}]
data.collect()  # returns a dict of Numpy arrays: {'a': array([0, .., 9]), 'b': array([10, ..., 19])}
```

If the dataset is made infinite, e.g. by calling `repeat()`, using a generator would be appropriate. Because `Dataset` is an iterable and generator is a special case of iterator, an `iter()` call is needed to get a generator that iterates over the items.

```python
data = # a Dataset instance ...
model = # build a Keras model ...
model.fit_generator(iter(data), ...)
```

`Dataset.tensorflow()` converts the dataset to a `tf.data.Dataset` object that can be further plugged into a TensorFlow pipeline:

```python
import numpy as np
import tensorflow as tf
from flazy import Dataset

dataset: tf.data.Dataset = Dataset(feature=np.random.randn(3, 3), label=[1, 2, 3]).tensorflow()

with tf.Session() as sess:
  input_op = dataset.make_one_shot_iterator().get_next()
  print(sess.run(input_op))
  print(sess.run(input_op))
  print(sess.run(input_op))

# outputs:
#     {'feature': array([0.45149828, 0.25305908, 1.47109173]), 'label': 1}
#     {'feature': array([-1.70763778, -1.1286721 ,  0.31254167]), 'label': 2}
#     {'feature': array([0.1586513 , 0.73838646, 1.57444576]), 'label': 3}          
```


Future Work
-----------

A non-exhaustive list of planned work:

* reader/writers for tfrecords, npz, common image/audio formats, hdf5, etc.
* more stable multithreading/multiprocessing
* examples in examples directory
* documentation website
* more extensive test suites and coverage measurement
