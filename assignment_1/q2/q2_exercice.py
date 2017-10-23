"""
Problem 2: Logistic regression
You can choose to do one of the three following tasks:
Task 1: Improving the accuracy of our logistic regression on MNIST (objective 97 % accuracy on test set)

"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training import training_util
from tensorflow.contrib.data import Dataset, TFRecordDataset

from functools import partial
import time

tf.logging.set_verbosity(tf.logging.INFO)   # require to see the training hooks stuff

def load_mnist(path='./data'):
    mnist = input_data.read_data_sets(path, one_hot=True)
    return mnist

def perso_mnist_net_model_fn(features, labels, mode=tf.estimator.ModeKeys.TRAIN, params=None):

    """
    a model_fn for Estimator class

    This function will be called to create a new graph each time an estimator method is called on an estimator using
    this function.
    """
    learning_rate = params['learning_rate']

    n_classes = 10

    #X = features['feature']
    X = features
    labels = tf.reshape(labels, [-1, n_classes])  # ou tf.squeeze
    Y = tf.cast(labels, tf.int32)
    flat_input = tf.keras.layers.Flatten()(X)
    fc1 = tf.keras.layers.Dense(100)(flat_input)
    fc2 = tf.keras.layers.Dense(100)(fc1)

    logits = tf.keras.layers.Dense(n_classes)(fc2)


    with tf.name_scope('predictions'):
        predictions = tf.argmax(logits, axis=1)

    with tf.name_scope('loss'):
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=logits)
        loss = tf.reduce_mean(entropy, name='loss') # computes the mean over all the examples in the batch
        #loss = tf.reduce_mean(-tf.reduce_sum(tf.cast(labels, tf.float32)*tf.log(logits), reduction_indices=1))


    gt_label = tf.argmax(Y, axis=1)
    accuracy = tf.contrib.metrics.accuracy(predictions, gt_label)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=training_util.get_global_step())
    #train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=training_util.get_global_step())

    #print("Declaring {}".format("_".join([mode, "loss"])))
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy_' + mode, accuracy)

    #images = tf.reshape(X, shape=[-1, 28, 28, 1])
    #tf.summary.image('input', images, max_outputs=32)
    #tf.summary.scalar("_".join([mode, "loss"]), accuracy)

    #logging_hook = tf.train.LoggingTensorHook({"accuracy":accuracy, "loss": loss}, every_n_iter=1, formatter=lambda x: '--- > logging hook {}'.format(x))
    logging_hook = tf.train.LoggingTensorHook({"loss":loss, "accuracy": accuracy}, every_n_iter=1, formatter=lambda x: '--- > logging hook {}'.format(x))
    #validation_hook = tf.train.LoggingTensorHook({"gt":gt_label, 'prediction': predictions, 'accuracy': accuracy}, every_n_iter=1, formatter=lambda x: '--- > logging hook {}'.format(x))
    validation_hook = tf.train.LoggingTensorHook({'accuracy': accuracy}, every_n_iter=1, formatter=lambda x: '--- > logging hook {}'.format(x))

    accuracy_metric = tf.contrib.metrics.streaming_accuracy(predictions, gt_label)

    log_hook = tf.train.LoggingTensorHook(
        {
            'loss': loss,
            'accuracy': accuracy,
            'step': training_util.get_global_step()
        },
        every_n_iter=100)


    return tf.estimator.EstimatorSpec(predictions=predictions, loss=loss, train_op=train_op, mode=mode,
                                      training_hooks=[log_hook], evaluation_hooks=[], eval_metric_ops={'acc_validation': accuracy_metric})

class TimerHook(tf.train.SessionRunHook):
    def before_run(self, run_context):
        self.start_at = time.time()

    def after_run(self, run_context, run_values):
        duration = time.time() - self.start_at
        #print("Duration of run was {}".format(duration))


#class MyNStepHook(tf.train.SessionRunHook):
#    def before_run(self, run_context):
#        tf.train.SessionRunArgs({'global_step': global_step, 'train_accuracy': train_accuracy})
#
#
#    def after_run(self, run_context, run_values):
#        if run_context.get
#        duration = time.time() - self.start_at
#        print("Duration of run was {}".format(duration))

def get_inputs_fn_based_on_generator():
    mnist = load_mnist()
    batch_size = 10
    from tensorflow.contrib.learn.python.learn.learn_io import generator_input_fn


    def input_fn(dataset, max_epoch=None):
        init_epochs = dataset.epochs_completed
        while True:
            if max_epoch is not None and dataset.epochs_completed - init_epochs >= max_epoch:
                raise StopIteration
            else:
                val = dataset.next_batch(1)
                yield {'feature': val[0], 'label': val[1]}


    my_input_fn_val = generator_input_fn(lambda: input_fn(mnist.validation, 1), target_key='label', batch_size=128)
    my_input_fn_test = generator_input_fn(lambda: input_fn(mnist.test, 1), target_key='label', batch_size=128)
    my_input_fn_train = generator_input_fn(lambda: input_fn(mnist.train, 30), target_key='label', batch_size=128)
    return my_input_fn_train, my_input_fn_val, my_input_fn_test

def get_input_fn_dataset(dataset_name = 'train', num_epoch=30, batch_size=128):
    # idea taken from https://stackoverflow.com/questions/45620449/initialization-of-tf-contrib-data-iterator-with-tf-estimator
    # pas de vrai amelioration en terme de speed.. TODO: tester sans iterateur tout dans le meme graph
    # in order to have everythings in same graph
    init_hook = IteratorInitHook()
    def _parse_function(example_proto):
      features = {"image": tf.FixedLenFeature([784], tf.float32),
                  "label": tf.FixedLenFeature([10], tf.float32)}
      #parsed_features = tf.parse_single_example(example_proto, features)
      parsed_features = tf.parse_example(example_proto, features)
      return parsed_features["image"], parsed_features["label"]

    def input_fn():
        dataset = TFRecordDataset('data/{}.tfrecords'.format(dataset_name), compression_type='GZIP')
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(100)
        dataset = dataset.repeat(num_epoch)
        dataset = dataset.map(_parse_function)

        iterator = dataset.make_initializable_iterator()
        init_hook.iterator_init_op = iterator.initializer
        #iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
        #image, label = iterator.get_next()
        #return image, label
    return input_fn, init_hook
    #return iterator.get_next()


# attention using a lambda here.. you get something really slow..
    #input_fn_train = lambda: input_fn('data/train.tfrecords', 30)
    #input_fn_val = lambda: input_fn('data/validation.tfrecords', 1)
    #input_fn_test = lambda: input_fn('data/validation.tfrecords', 1)

    #input_fn_train = partial(input_fn, 'data/train.tfrecords', 30)
    #input_fn_val = partial(input_fn, 'data/validation.tfrecords', 1)
    #input_fn_test = partial(input_fn, 'data/validation.tfrecords', 1)






class IteratorInitHook(tf.train.SessionRunHook):

    def after_create_session(self, session, coord):
        session.run(self.iterator_init_op)


def main():

    config= tf.estimator.RunConfig()
    config = config.replace(save_summary_steps=10000)
    config = config.replace(save_checkpoints_steps=10000)
    config = config.replace(keep_checkpoint_max=20)

    if tf.app.flags.FLAGS.enable_tfrecords:
        input_train, input_train_init_hook = get_input_fn_dataset('train', 30)
        input_validation, input_validation_init_hook = get_input_fn_dataset('validation', 1)
        input_test, input_test_init_hook = get_input_fn_dataset('test', 1)
    else:
        input_train, input_validation, input_test = get_inputs_fn_based_on_generator()

    print("Test")

    params = {'learning_rate': 0.01, 'dropout': 1.00}

    if tf.app.flags.FLAGS.enable_estimators:
        estimator = tf.estimator.Estimator(model_fn=perso_mnist_net_model_fn,
                                           model_dir='mnist_test',
                                           params=params,
                                           config=config)

        estimator.train(input_fn=input_train, hooks=[input_train_init_hook])
    else:
        with tf.Session() as sess:
            next_batch = input_train()
            X = next_batch[0]
            Y = next_batch[1]
            estimator_spec = perso_mnist_net_model_fn(X, Y, params=params)

            sess.run(input_train_init_hook.iterator_init_op)
            sess.run(tf.global_variables_initializer())
            while True:
                start = time.time()
                loss_val = sess.run(estimator_spec.train_op)
                duration = time.time() - start
                print("Nb/sec = {}".format(1.0/duration))


    #print("Bench iterator speed")
    #session = tf.Session()
    #session.run(input_train_init_hook.iterator_init_op)
    #while True:
    #    start = time.time()
    #    val = session.run(input_train())  # <-- the duration never stop to increase here
    #    #print(val)

    #    duration = time.time() - start
    #    print("duration is {}".format(duration))


if __name__ == "__main__":
    tf.app.flags.DEFINE_boolean("enable_tfrecords", True, """True = use tfrecords, False = use python generator (slow)""")
    tf.app.flags.DEFINE_boolean("enable_estimators", True, """True = use estimators""")
    main()