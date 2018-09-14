"""
Profile and debug tf

Train a simple DNN to see how tf.profiler and tfdbg works with the high level estimator
Modified from work by Aymeric Damien @https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function
import logging
import argparse
import os.path

import tensorflow as tf

from tfpd import constants

logger = logging.getLogger('tensorflow')

PROFILE_FILENAME = 'profile.pb'


def neural_net(x_dict, num_classes, activation_fn, num_layers):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['image']
    # Hidden fully connected layer with 256 neurons

    prev_layer = x

    for i in range(num_layers):
        prev_layer = tf.layers.dense(prev_layer, 256, activation=activation_fn)

    out_layer = tf.layers.dense(prev_layer, num_classes)
    return out_layer


# Define the model function (following TF Estimator Template)
def create_model_fn(learning_rate, num_classes, activation, num_layers):
    """
    Creates a model function
    :param learning_rate:
    :param num_classes:
    :param activation:
    :return: model_fn of type (features_dict, labels, mode) -> :class:`tf.estimator.EstimatorSpec`
    """

    def fn(features, labels, mode):
        # Build the neural network
        if activation:
            activation_fn = tf.nn.relu
        else:
            activation_fn = None

        logits = neural_net(features, num_classes, activation_fn, num_layers)

        # Predictions
        pred_classes = tf.argmax(logits, axis=1)
        pred_probas = tf.nn.softmax(logits)

        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(pred_probas)
        }

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes, export_outputs=export_outputs)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op},
            export_outputs=export_outputs
        )

        return estim_specs

    return fn


def create_input_fn(images, labels, batch_size, num_epochs=None, shuffle=False, queue_capacity=1000, num_threads=1):
    """
    Creates an input function that feeds data to a model
    :param images:
    :param labels:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :param queue_capacity:
    :param num_threads:
    :return: input_fn of type () -> features_dict
    """
    return tf.estimator.inputs.numpy_input_fn(
        x={'image': images}, y=labels,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        queue_capacity=queue_capacity,
        num_threads=num_threads
    )


def feature_spec():
    return {
        'image': tf.FixedLenFeature([constants.MNIST_INPUT_DIM], tf.float32),
        'label': tf.FixedLenFeature([], tf.int64)
    }


def build_features(src_paths, num_read_threads, num_parse_threads, queue_capacity, batch_size, epochs=None,
                   shuffle=False):
    # Both features and labels in our case are stored in the same tf.Example proto.
    from tensorflow.contrib.data import make_batched_features_dataset

    # A lot happens here, we:
    # * read tf.Examples stored in tf.Records
    # * parse Examples using features spec from normalized_feature_spec
    # * we parallelize reading and parsing
    # * we batch and repeat whole training dataset epoch times
    # interleaving is applied whenever num_read_threads > 1

    def fn():
        d = make_batched_features_dataset(src_paths,
                                          batch_size=batch_size,
                                          num_epochs=epochs,
                                          shuffle=shuffle,
                                          reader_num_threads=num_read_threads,
                                          parser_num_threads=num_parse_threads,
                                          prefetch_buffer_size=queue_capacity,
                                          sloppy_ordering=True if num_read_threads > 1 else False,
                                          features=feature_spec())

        # `neural_net` expects to be fed a tuple of (features, labels)
        return d.map(lambda x: ({'image': x['image']}, x['label']))

    return fn


def parse_args():
    """
    Parse cmd arguments
    :return: :class:`ArgumentParser` instance
    """
    arg_parser = argparse.ArgumentParser(description='MNIST Deep Neural Network')
    arg_parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    arg_parser.add_argument('--num-layers', type=int, default=2,
                            help='Use this to create a deep model, so you can see trade-offs in compute vs IO')
    arg_parser.add_argument('--num-epochs', type=int, default=1, help='Num training epochs')
    arg_parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    arg_parser.add_argument('--queue-capacity', type=int, default=100, help='Capacity for the reading queue')
    arg_parser.add_argument('--num-read-threads', type=int, default=1, help='Number of threads for reading data')
    arg_parser.add_argument('--num-parse-threads', type=int, default=1, help='Number of threads for parsing data')
    arg_parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
    arg_parser.add_argument('--no-activation', dest='activation', action='store_false')
    arg_parser.add_argument('--model-dir', required=True, help='Path to model dir')
    arg_parser.add_argument('--train-data', required=False, default=None,
                            help='Path to input data path. If undefined, loads data into memory')
    arg_parser.add_argument('--debug', dest='debug', action='store_true', help='Run with tfdbg')
    arg_parser.set_defaults(activation=True, shuffle=True, debug=False)

    args = arg_parser.parse_args()
    logger.info("Running with args:")
    for arg in vars(args):
        logger.info("\t%s: %s", arg, getattr(args, arg))

    return args


def main():
    """
    Runs training and testing of mnist on a pre-defined neural network
    :return:
    """
    args = parse_args()

    if args.train_data:
        # use external data, generated from :mod:`tfpd.data`
        trn_files = tf.gfile.Glob(os.path.join(args.train_data, 'train', '*.tfrecords'))
        tst_files = tf.gfile.Glob(os.path.join(args.train_data, 'test', '*.tfrecords'))

        trn_input_fn = build_features(trn_files,
                                      num_read_threads=args.num_read_threads,
                                      num_parse_threads=args.num_parse_threads,
                                      queue_capacity=args.queue_capacity,
                                      batch_size=args.batch_size,
                                      epochs=args.num_epochs,
                                      shuffle=args.shuffle)
        tst_input_fn = build_features(tst_files,
                                      num_read_threads=args.num_read_threads,
                                      num_parse_threads=args.num_parse_threads,
                                      queue_capacity=args.queue_capacity,
                                      batch_size=args.batch_size,
                                      epochs=args.num_epochs,
                                      shuffle=args.shuffle)

    else:
        # load tf mnist
        from tensorflow.examples.tutorials.mnist import input_data
        import tempfile
        mnist = input_data.read_data_sets(os.path.join(tempfile.gettempdir(), 'mnist'), one_hot=False)
        trn_input_fn = create_input_fn(
            mnist.train.images,
            mnist.train.labels,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            shuffle=args.shuffle,
            queue_capacity=args.queue_capacity,
            num_threads=args.num_read_threads
        )

        tst_input_fn = create_input_fn(
            mnist.test.images,
            mnist.test.labels,
            batch_size=args.batch_size,
            num_epochs=1,
            shuffle=args.shuffle,
            queue_capacity=args.queue_capacity,
            num_threads=args.num_read_threads
        )

    with tf.contrib.tfprof.ProfileContext(args.model_dir) as pctx:
        # -- options --
        # opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
        # pctx.add_auto_profiling('op', opts, profile_steps())

        # Build the Estimator
        model_fn = create_model_fn(learning_rate=args.lr,
                                   num_classes=constants.MNIST_OUTPUT_DIM,
                                   activation=args.activation,
                                   num_layers=args.num_layers)
        model = tf.estimator.Estimator(model_fn, args.model_dir)

        # Train the Model
        hooks = []
        if args.debug:
            from tensorflow.python import debug as tf_debug
            hooks.append(tf_debug.LocalCLIDebugHook())

        model.train(trn_input_fn, hooks=hooks)

        # Use the Estimator 'evaluate' method
        e = model.evaluate(tst_input_fn)

        logger.info("Testing Accuracy: %s", e['accuracy'])

        with open(os.path.join(args.model_dir, PROFILE_FILENAME), 'wb') as fp:
            fp.write(pctx.profiler.serialize_to_string())
            logger.info("Saved profile to %s", args.model_dir)


if __name__ == '__main__':
    main()
