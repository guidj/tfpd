import argparse
import os
import os.path
import logging

import tensorflow as tf
import numpy as np

from tfpd import constants

logger = logging.getLogger('tensorflow')


def example():
    """
    Creates a fake MNIST example
    Values in the range [0, 1) are split into constants.MNIST_OUTPUT_DIM bins, and the label is defined by the
    membership of the first pixel to a bin
    :return:
    """
    image = np.random.rand(constants.MNIST_INPUT_DIM)
    label = np.random.randint(0, 10)
    for i in range(constants.MNIST_OUTPUT_DIM):
        image[i] = 0
    image[label] = 1

    feature = {
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'image': tf.train.Feature(float_list=tf.train.FloatList(value=image.tolist()))
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_args():
    arg_parser = argparse.ArgumentParser(description='Generate random data for training/testing')
    arg_parser.add_argument('--n', required=False, default=1000000, type=int, help='Number of records to generate')
    arg_parser.add_argument('--partitions', required=False, default=100, type=int, help='Number of partitions')
    arg_parser.add_argument('--output', required=True, type=str, help='Output path')

    return arg_parser.parse_args()


def data_path(path, subdir, part, n_parts):
    path = os.path.join(path, subdir, 'part-{}-of-{}.tfrecords'.format(part, n_parts))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    return path


def main():
    args = parse_args()

    logger.info('Generating training/testing data at {}'.format(args.output))
    logger.info('{} examples, {} partitions'.format(args.n, args.partitions))

    block_size = args.n / args.partitions

    partition = None
    train_writer = None
    test_writer = None

    for i in range(args.n):
        block = i / block_size + 1

        if partition != block:
            partition = block
            train_writer = tf.python_io.TFRecordWriter(data_path(args.output, 'train', partition, args.partitions))
            test_writer = tf.python_io.TFRecordWriter(data_path(args.output, 'test', partition, args.partitions))

            logger.info('Writing partition {}/{}'.format(partition, args.partitions))

        train_writer.write(example().SerializeToString())
        test_writer.write(example().SerializeToString())

    if args.n > 0:
        train_writer.close()
        test_writer.close()

    logger.info('Done')


if __name__ == '__main__':
    main()
