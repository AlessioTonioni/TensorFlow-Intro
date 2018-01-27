
# coding: utf-8

# ## 1)Loading input data
# We will use placeholder for this sample

# In[ ]:


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#download mnist dataset
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

#create a placeholder with two dimension
#the first one is yet unknown the second one is the number of features in mnist i.e. 28x28 image
x = tf.placeholder(tf.float32, [None,784])

#create a placeholder for the labels
y = tf.placeholder(tf.int64, [None,10])


# ## 2)Model Definition
# Softmax Regressor

# In[ ]:


#define variables
W = tf.Variable(tf.zeros([784,10])) #784 inputs, 10 nodes as output => 1 vs all classification
b = tf.Variable(tf.zeros([10]))

#model prediction
prediction_logits = tf.matmul(x,W)+b
prediction = tf.nn.softmax(prediction_logits)


# 2 layer neural network

# In[ ]:


W_1 = tf.get_variable("w_1",shape=[784,50],initializer=tf.contrib.layers.xavier_initializer())  #784 inputs, 50 nodes as output => 50 hidden units
b_1 = tf.get_variable("b_1",shape=[50],initializer=tf.random_uniform_initializer(-0.5,0.5))

#fully connected layer 1
fc1 = tf.nn.relu(tf.matmul(x,W_1)+b_1)  #relu activation

W_2 = tf.get_variable("w_2",shape=[50,10],initializer=tf.contrib.layers.xavier_initializer())
b_2 = tf.get_variable("b_2",shape=[10],initializer=tf.random_uniform_initializer(-0.5,0.5))

#model prediction
prediction_logits = tf.matmul(fc1,W_2)+b_2
prediction = tf.nn.softmax(prediction_logits)


# ## 3)Loss function definiton
# Both the model use cross entropy as loss function

# In[ ]:


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction_logits, name='cross_entropy'))


# ## 4)Create minimizer and train op

# In[ ]:


global_step = tf.Variable(0, trainable=False, name="global_step")         #variable to keep track of the steps
lr = 0.01                                                                 #learning rate

train_op = tf.train.MomentumOptimizer(lr,0.90).minimize(loss, global_step=global_step)


# We also want to evaluate the performance of the learned model

# In[ ]:


correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))   #array of bool
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ## 5) Create session and train model
# We will use an interactive session for this sample

# In[ ]:


sess=tf.InteractiveSession()


# Iniatialize the variables

# In[ ]:


sess.run(tf.global_variables_initializer())


# train for 10000 step with batchsize 128

# In[ ]:


for i in range(10000):
    batch_x, batch_y = mnist.train.next_batch(128)
    _,loss_value = sess.run([train_op,loss], feed_dict={x: batch_x, y: batch_y})
    if i%10 is 0:
        print('Step {}/10000, loss: {}'.format(i,loss_value))


# Test the model

# In[ ]:


acc_score = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
print('Model accuracy after 1000 step: {}'.format(acc_score))


# Visualize an image with the prediction

# In[ ]:


get_ipython().magic('matplotlib inline')
img_batch,pred = sess.run([x,prediction], feed_dict={x:batch_x,y:batch_y})
img = img_batch[0]
img = np.reshape(img,(28,28))
plt.imshow(img, cmap='gray')
print('Raw predictions: {}'.format(pred[0]))
print('Class predicted {}'.format(np.argmax(pred[0])))


# Close Session

# In[ ]:


sess.close()

