"""
Just a script to convert to a format supported by the new dataset API
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tqdm

def to_tfrecord(input_fn, output_fname):
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(output_fname, options=options)
    print("Writting file {}".format(output_fname))

    for i in tqdm.tqdm(range(input_fn.num_examples)):
        image, label = input_fn.next_batch(1)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(float_list=tf.train.FloatList(value=image.flatten().tolist())),
            'label': tf.train.Feature(float_list=tf.train.FloatList(value=label.flatten().tolist()))
        }))
        writer.write(example.SerializeToString())
    writer.close()


def main():

    path = './'
    mnist = input_data.read_data_sets(path, one_hot=True)

    to_tfrecord(mnist.train, 'train.tfrecords')
    to_tfrecord(mnist.validation, 'validation.tfrecords')
    to_tfrecord(mnist.test, 'test.tfrecords')



if __name__ == "__main__":
    main()
