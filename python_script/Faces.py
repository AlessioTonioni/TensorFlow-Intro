
# coding: utf-8

# # Jap Idol challenge
# Try to correctly classify images depicting faces of 40 famous japanese idol. The dataset is composed of 7200 112x112  rgb image divided into 6 different tfrecords file. 
# Unfortunatelly I have not been able to trace back the original author of the dataset, if you know him let me know and I'll had credits where are due.

# ## Graph Definition

# ### 1)Input reading
# Craete symbolic ops to read single examples from train and validation set, then create minibatch with them

# In[1]:


import tensorflow as tf
import os
import time
from datetime import datetime

def read_and_decode_tfrecords(filenames,image_shape):
    """
    Read and decode tfrecords from the list of files filenames, returns a single image and label as tensorflow op
    """
    
    #create filename queue
    filename_queue = tf.train.string_input_producer(filenames)
    
    #symbolic reader to read one example at a time
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    #read the fields of the single example
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([1], tf.int64),
        }
    )

    #decode the image data
    image = tf.image.decode_jpeg(features['image_raw'], channels=image_shape[-1])
    image.set_shape(image_shape)
    
    #convert to float --> tensorflow works only on float!
    image = tf.cast(image, tf.float32)
    
    #compress range between [-1,1]
    image = tf.div(image,127.5)
    image = tf.subtract(image,1)
    
    return image,features['label']

#Params
IMAGE_SHAPE = [112,112,3]
BATCH_SIZE = 64

#get_filenames
cwd = os.path.join(os.getcwd(),'data','faces')
tf_records_filenames = [os.path.join(cwd,f) for f in os.listdir(cwd)]
training_set = tf_records_filenames[:-1] #use 5 out of 6 tfrecord as training set
validation_set = [tf_records_filenames[-1]]   #use 1 out of 6 tfrecord as validation set

print('Going to train on: {}'.format(training_set))
print('Going to validate on: {}'.format(validation_set))

with tf.variable_scope('train_reader'):
    #create single example op
    image_t,label_t = read_and_decode_tfrecords(training_set,IMAGE_SHAPE)
    #create minibatch 
    image_batch_t,label_batch_t = tf.train.shuffle_batch([image_t,label_t], batch_size=BATCH_SIZE, num_threads=8, capacity=1000, min_after_dequeue=300)
    #create image summaries to diplay training image in tensorflow
    tf.summary.image("train_images",image_batch_t,max_outputs=5)
    
with tf.variable_scope('val_reader'):
    image_v,label_v = read_and_decode_tfrecords(validation_set,IMAGE_SHAPE)
    image_batch_v, label_batch_v = tf.train.shuffle_batch([image_v,label_v], batch_size=BATCH_SIZE, num_threads=8,capacity=1200, min_after_dequeue=600)
    tf.summary.image("validation_image",image_batch_v,max_outputs=5)


# ### 2)Model Definition
# Define a CNN model to predict one of the 40 possible classes, add summary op to visualize useful statistic in tensorboard.  

# In[2]:


#Some utility function 

def variable_summaries(var, name):
    """
    Create image summaries and histogram for var.
    """
    #summary operation must be run on cpu
    with tf.device("/cpu:0"):
        with tf.name_scope("summaries"):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

def conv(previous_layer, shape, stride):
    kernel = tf.get_variable("kernel", shape=shape,trainable=True, initializer=tf.contrib.layers.xavier_initializer())
    variable_summaries(kernel,"kernel")

    bias = tf.get_variable("bias", shape=[shape[-1]],trainable=True,initializer=tf.random_uniform_initializer(-0.5,0.5))
    variable_summaries(bias,'bias')
    
    #conv +bias
    conv = tf.nn.conv2d(previous_layer,kernel,stride,padding='SAME')
    bias = tf.nn.bias_add(conv,bias)
    return bias

def fc(previous_layer, num_nodes):
    input_num_features = previous_layer.get_shape()[-1]
    weight_shape = [input_num_features,num_nodes]
    
    weights = tf.get_variable('weights',shape=weight_shape, trainable=True, initializer=tf.contrib.layers.xavier_initializer())
    variable_summaries(weights,'weights')
    
    bias = tf.get_variable('bias',shape=[num_nodes], trainable=True, initializer=tf.random_uniform_initializer(-0.5,0.5))
    variable_summaries(bias,'bias')
    
    return (tf.matmul(previous_layer,weights)+bias)

def getModel(image,num_classes):
    net = image
    input_channels = image.get_shape()[-1].value
    
    with tf.variable_scope('conv_1') as scope:
        #32 5x5 convolutional filters + relu activation
        net = conv(net,[5,5,input_channels,32],[1,1,1,1])
        net = tf.nn.relu(net)
    
    with tf.variable_scope('pool1') as scope:
        #max_pooling to halve dimensions
        net = tf.nn.max_pool(net,[1,2,2,1],[1,2,2,1],padding='SAME')
   
    with tf.variable_scope('conv_2') as scope:
        #64 3x3 convolutional filters + relu activation
        net = conv(net,[3,3,32,64],[1,1,1,1])
        net = tf.nn.relu(net)
    
    with tf.variable_scope('pool_2') as scope:
        #max pooling to halve dimensions
        net = tf.nn.max_pool(net,[1,2,2,1],[1,2,2,1],padding='SAME')
    
    with tf.variable_scope('conv_3') as scope:
        #128 3x3 convolutional filters+relu
        net = conv(net,[3,3,64,128],[1,1,1,1])
        net=tf.nn.relu(net)
    
    with tf.variable_scope('pool_3') as scope:
        #max pooling to halve dimensions
        net = tf.nn.max_pool(net,[1,2,2,1],[1,2,2,1],padding='SAME')
    
    with tf.variable_scope('conv_4') as scope:
        #64 3x3 convolutional filters
        net = conv(net,[3,3,128,64],[1,2,2,1])
        net = tf.nn.relu(net)
    
    with tf.variable_scope('fc5') as scope:
        #256 fully connected node + relu
        linear_input_size = (net.get_shape()[-2].value**2) * net.get_shape()[-1].value
        #linearize feature maps
        net = tf.reshape(net,[BATCH_SIZE,linear_input_size])
        
        net = fc(net, 256)
        net = tf.nn.relu(net)
    
    with tf.variable_scope('fc6') as scope:
        #40 output fully connected node
        net = fc(net,num_classes)
    
    return net


# In[3]:


#Create two graph, one for training one for evaluation
with tf.variable_scope('Network') as scope:
    
    #graph that takes as input the training batch
    prediction_train = getModel(image_batch_t,40)
    train_classification = tf.nn.softmax(prediction_train)
    print('Train model ready')
    
    #enable sharing of variables between models
    scope.reuse_variables()
    
    #graph that takes as input the validation batch
    prediction_val = getModel(image_batch_v,40)
    validation_classification = tf.nn.softmax(prediction_val)
    print('Validation model ready')


# ### 3)Loss Function
# We are going to use once again cross entropy as loss function

# In[4]:


with tf.variable_scope('loss') as scope:
    label_batch_t = tf.cast(label_batch_t, tf.int64)
    label_batch_t = tf.reshape(label_batch_t,[BATCH_SIZE])

    #our labels are not one hot encoded --> use sparse_softmax_cross_entropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch_t, logits=prediction_train, name='cross_entropy')
    loss = tf.reduce_mean(cross_entropy)

    #add summary op
    tf.summary.scalar('loss', loss)


# ### 4)Optimizer
# Define a suitable optimizer and learning rate

# In[5]:


learning_rate = 0.0001
global_step = tf.Variable(0, trainable=False, name="global_step")

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)


# ### 5)Accurcay Op
# Define an accuracy op to measure how well the model performs

# In[6]:


with tf.variable_scope('accuracy') as scope:
    label_batch_v = tf.cast(label_batch_v, tf.int64)
    label_batch_v = tf.reshape(label_batch_v,[BATCH_SIZE])

    correct_prediction = tf.equal(label_batch_v,tf.argmax(prediction_val,1))   #array of bool
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('validation_accuracy',accuracy)


# ### 6)Meta op
# Add utility op to save summaries and model

# In[7]:


#init operation
init = tf.global_variables_initializer()

#summary operation for visualization
summary_op = tf.summary.merge_all()

#create a saver to save and restore models
saver = tf.train.Saver(tf.global_variables(),max_to_keep=2)


# ## Training 

# In[ ]:


log_folder = os.path.join('log','faces')

#create log folder if it does not exist
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

    
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    max_step = 10000
    step = 0
    
    #create summary writers
    sum_writer = tf.summary.FileWriter(log_folder,sess.graph)
    
    #check if there is a weight checkpoint file in log_folder 
    ckpt = tf.train.get_checkpoint_state(log_folder)
    if ckpt and ckpt.model_checkpoint_path:
        #if found restore saved model --> useful to split training in different moment of time
        print('Founded valid checkpoint file at: '+ckpt.model_checkpoint_path)

        #restore variable
        saver.restore(sess, ckpt.model_checkpoint_path)

        #restore step
        step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print('Restored %d step model'%(step))
    else:
        #if no checkpoint file is found
        print('No checkpoint file found, initialization instead')
        
        #init all variables
        sess.run(init)

    #start input threads 
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    #start training procedure
    for i in range(step,max_step):
        #every step
        start_time = time.time()
        _,loss_value = sess.run([train_op,loss])
        duration = time.time() - start_time
        
        if step%10 is 0:
            #every 10 step evaluate on a validation batch and print info
            validation_score = sess.run(accuracy)
            
            num_examples_per_step = BATCH_SIZE
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            
            format_str = ('%s: step %d, loss = %.2f, validation_accuracy=%.2f,(%.1f examples/sec; %.3f sec/batch)')
            print (format_str % (datetime.now(), step, loss_value,validation_score,examples_per_sec, sec_per_batch))
        
        if step%50 is 0:
            #every 50 step save summaries
            summary_str = sess.run(summary_op)
            sum_writer.add_summary(summary_str, step)
            
        if step%1000 is 0:
            #every 1000 step save checkpoint files with all the weights
            checkpoint_path = os.path.join(log_folder, 'faces.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            
        step+=1
    #shut down all the threads
    coord.request_stop()

print('ALL DONE!')

