# Neural Network Handwriting Recognition using TensorFlow
/**
*
* @author ηεεтεsн                                                 
*
*/



from __future__ import print_function
import tensorflow as tf

#import dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input.read_data_sets('/temp/data/',  one_hot = True)

#hyperparameters
learning_rate = 0.001 #tradeoff between speed and accuracy
training_iters = 200000 #training iterations
batch_size = 128 #The size of our dataset (Our sample size)
display_step = 10 #display what's happening at every 10 iterators

#network parameters
#28 x 28 image size
n_input = 784
n_classes = 10 #(Since we are training for detecting digits, thus the no. of digits are 10, ie from 0 to 9. Thus we are using 10 classes)
droupout = 0.75 #(awesome technique, randomly turn offs the neurons while the data is flowing through. This is done so that the neural network searches for new pathways and it becomes fit for more generalized datasets. This is like tuning our neural network)

#Two gateways for our data. One is for the images, and the other for the labels.Thus our neural network will be seeing two datasets at the same time. One will be the image (Like an image of no. 6), and the other will be the label which we have fed it to train, like the no. '6' in actual. float32 bit means the gateway is of 32 bits or 4 bytes float
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)


#convolutional layer (convolution means transform)
#It takes the image and processes it and tranforms it into something simple
def conv2d(x, W, b, strides=1):
    #strides are tensors and tensors means data
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides,1], padding='SAME')
    x = tf.nn.bias_add(x, b) #makes model accurate
    return tf.nn.relu(x) #relu (rectified linear unit) is an activation function

#pooling takes small square samples from the image produced by the convolutional layer and processes them and it produces the single output for them
#max pooling means we wanna take the maximum of the learning patterns of the network
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize = [1,k,k,1], strides = [1,k,k,1], padding = 'SAME')
#ksize and strides are 4 d tensors since they both have 4 variables

#create model
#The x is our input, the weights are the connections between our layers and biases effect our layers for accuracy and droupout is for generalization
def conv_model(x, weights, biases, droupout):
    #reshape input
    x = tf.reshape(x, shape = [-1, 28, 28, 1]) #resizing our image and then we'll call our convolutional model

    #Ist LAYER convolutional layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1']) #we are using the convolutional function we defined earlier. conv1 is our convolution
    #max pooling. We are taking conv conv1 as a parament and adding maxpooling to it and storing in the convolution again
    conv1 = maxpool2d(conv1, weights['wc2'], biases['bc2'])

    #IInd LAYER (It takes first layer as input)
    conv2 = conv2(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2) #max pooling it too

    #FULLY CONNECTED LAYER, a generic layer. Every neuron in this layer is connected to the evry neuron in the previous layer. This layer is just the representation of the transformation which the previous layers have carried out.
    #reshaping data
    fc1 =tf.reshape(conv2, [-1, weights['wd1'.get_shape().as_list()]])
    #matrix multiplication to combine data so far
    fc1 = tf.add(tf.matmul(fc1, weights['wd1'], biases['bd1']))
    fc1 = tf.nn.relu(fc1)
    #applying droupout
    fc1 = tf.nn.dropout(fc1, droupout)

    #output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out', biases['out']]))

#CREATE weights (A dictionary, a list of 4 tensorflow variables)
weights = {
'wc1': tf.variable(tf.random_normal([5,5,1,32])), #first set of weights, This is a 5 by 5 convolution with 1 input (an image) and 32 outputs (bits)
'wc2': tf.variable(tf.random_normal([5,5,32,64])), #second set of weights, This is a 5 by 5 convolution with 32 inputs (different connections) and 64 outputs (bits) convolutional layer
'wd1': tf.variable(tf.random_normal([7*7*64,1024])), #Third set of weights, 7*7*64 inputs and 1024 outputs (bits)
'out': tf.variable(tf.random_normal([1024,n_classes])), #last set of weights, 1024 inputs and no. of classes (10)
}

#construct model using our i/p data x, with weights and biases and our keep_prob which is our droupout
pred = conv_net(x, weights, biases, keep_prob)

#define optimizer and loss
cost = tf.reduce_mean(tf.nnsoftmax_cross_entropy_with_logits(pred, y)) #It measures the probability error and the classication tasks. This means that the classes are mutually exclusive ie, the images with 0 and the image with 1 are exclusive

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost) #The more we minimize our cost the more accurate our system is

#evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) #we have the correct data and we have the predicition which we can use to do measure our accuracy
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#initialize
init = tf.initialize_all_variables()

#launch the graph
with tf.session as sess:
    sess.run(init)
    step = 1
    #keep training until max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))
