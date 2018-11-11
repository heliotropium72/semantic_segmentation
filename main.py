#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# Hyperparameters
# Some of these values have to be feed to the right placeholder of the vgg model
EPOCHS = 30
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
KEEP_PROB = 0.7
REG = 1e-3 # Strength of L2 regularization (punishs if one parameter gets much more used than others)
STD = 0.01 # Standard deviation for weight initialisation (normal distribution aroud zero with width STD)

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)    
    return w1, keep, l3, l4, l7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Regularization to avoid overfitting
    reg = tf.contrib.layers.l2_regularizer(REG)
    # Weight initialisation of new layer to introduce asymetry to the learning process
    init=tf.truncated_normal_initializer(stddev=STD)
    # Padding "same" is crucial to get same size

    # Layer 7
    #1x1 convolution
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
                                padding='same',
                                kernel_regularizer=reg,
                                kernel_initializer=init)
    # transpose convolution
    output = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, 2, padding='same',
                                        kernel_regularizer=reg,
                                        kernel_initializer=init)
    
    # Add some skip layers
    # Layer 4
    l4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                   kernel_regularizer=reg,
                                   kernel_initializer=init)
    output = tf.add(output, l4_conv_1x1)
    output = tf.layers.conv2d_transpose(output, num_classes, 4, 2, padding='same', 
                                        kernel_regularizer=reg,
                                        kernel_initializer=init)
    
    # Layer 3
    l3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                   kernel_regularizer=reg,
                                   kernel_initializer=init)
    output = tf.add(output, l3_conv_1x1)
    output = tf.layers.conv2d_transpose(output, num_classes, 16, 8, padding='same', 
                                        kernel_regularizer=reg,
                                        kernel_initializer=init)
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    # Minimize cross entropy and the l2 regularization defined in the graph
    total_loss = cross_entropy_loss + tf.losses.get_regularization_loss()
    # optimizer operations
    loss_operation = tf.reduce_mean(total_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    training_operation = optimizer.minimize(loss_operation)
    return logits, training_operation, cross_entropy_loss
tests.test_optimize(optimize)

def evaluate(nn_last_layer, correct_label, num_classes):
    """ This is not yet used """
    tf_metric, tf_metric_update = tf.metrics.mean_iou(correct_label,
                                                      nn_last_layer,
                                                      num_classes,
                                                      name="IOU")
    #inside seesion
    #feed_dict={tf_label: labels[i], tf_prediction: predictions[i]}
    #session.run(tf_metric_update, feed_dict=feed_dict)
    # Calculate the score
    #score = session.run(tf_metric)
    #print("[TF] SCORE: ", score)
    return tf_metric

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images (from vgg model, needed for feed_dict)
    :param correct_label: TF Placeholder for label images (from vgg model, needed for feed_dict)
    :param keep_prob: TF Placeholder for dropout keep probability (from vgg model, needed for feed_dict)
    :param learning_rate: TF Placeholder for learning rate (from vgg model, needed for feed_dict)
    """

    # TODO: Implement function
    for epoch in range(epochs):
        total_loss = 0
        for images, label in get_batches_fn(batch_size):
            feed_dict = {input_image: images,
                         correct_label: label,
                         keep_prob: KEEP_PROB,
                         learning_rate: LEARNING_RATE}
            # Training
            sess.run(train_op, feed_dict=feed_dict)
            # Evaluation
            loss = sess.run(cross_entropy_loss, feed_dict=feed_dict)
            total_loss += loss
            #iou = sess.run(evaluate())
        print("EPOCH {}/{} : ".format(epoch+1, epochs) + \
              "Total loss = {:.2f}, ".format(total_loss) + \
              "batch loss = {:.3f}".format(loss))
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = '/data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        # input_image and keep_prob are placeholders
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        
        # Placeholder for variable resolution images
        # (the placeholders are filled with values during the training and loading of the images) 
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))
        learning_rate = tf.placeholder(dtype=tf.float32)
        
        logits, training_operation, cross_entropy_loss = optimize(nn_last_layer=layer_output, correct_label=correct_label,
                                                                  learning_rate=learning_rate, num_classes=num_classes)
        
        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs=EPOCHS, batch_size=BATCH_SIZE, get_batches_fn=get_batches_fn,
                 train_op=training_operation, cross_entropy_loss=cross_entropy_loss, input_image=input_image,
                 correct_label=correct_label, keep_prob=keep_prob, learning_rate=learning_rate)
        
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
        print('All finished.')

def make_movie():
    # Produce a little movie
    runs_dir = './runs'
    from glob import glob
    from os.path import join
    filenames = glob(join(runs_dir, '1541861859.5053282', '*.png'))
    filenames.sort()

    # Create video
    import imageio
    with imageio.get_writer(join(runs_dir, 'result.gif'), mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        
if __name__ == '__main__':
    run()
    #make_movie()
