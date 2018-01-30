
# coding: utf-8

# ## 1)Loading input data
# We will use placeholder for this sample

# In[ ]:


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#download mnist dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#create a placeholder with two dimension
#the first one is yet unknown the second one is the number of features in mnist i.e. 28x28 image
x = tf.placeholder(tf.float32, [None,784])

#create a placeholder for the labels
y = tf.placeholder(tf.int64, [None,10])


# ## 2)Model Definition

# In[ ]:


def get_linear_layer(x,num_output):
    #creates a linear layer with num_output outputs
    input_size = x.get_shape()[-1].value
    W = tf.get_variable("weights",shape=[input_size,num_output],initializer=tf.contrib.layers.xavier_initializer())  
    b = tf.get_variable("biases",shape=[num_output],initializer=tf.random_uniform_initializer(-0.5,0.5))
    
    return tf.matmul(x,W)+b

def get_fc_layer(x,num_neurons):
    #create a fully connected layer with num_neurons neurons
    return tf.nn.relu(get_linear_layer(x,num_neurons))


# Softmax Regressor

# In[ ]:


#model prediction
prediction_logits = get_linear_layer(x,10)
prediction = tf.nn.softmax(prediction_logits)


# 2 layer neural network

# In[ ]:


with tf.variable_scope('hidden_layer'):
    #fully connected layer 1
    fc1 = get_fc_layer(x,50)

with tf.variable_scope('output'):
    #model prediction
    prediction_logits = get_linear_layer(fc1,10)
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

