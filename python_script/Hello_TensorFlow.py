
# coding: utf-8

# # Algebraic Operations, graph and session
# 
# **Step 1**: import all the necesary packages

# In[ ]:


import numpy as np
import tensorflow as tf
print("Package Loaded")


# **step 2:** Create three tensorflow constants

# In[ ]:


a = tf.constant(1)
b = tf.constant(1)
c = tf.constant([[1,1],[1,1]])

print('a: {}'.format(a))
print('b: {}'.format(b))
print('c: {}'.format(c))


# **step 3:** Create some operations

# In[ ]:


op_1 = tf.add(a,b)
op_2 = tf.add(a,c) #a is added to each element of c --< numpy like broadcasting

print('op1: {}'.format(op_1))
print('op2: {}'.format(op_2))


# **step 4:** create a tensorflow session and evaluate tensors and ops

# In[ ]:


with tf.Session() as sess:
    a_val,b_val,c_val = sess.run([a,b,c])
    print('VARIABLES EVALUATED: ')
    print('a_val: Type: {} || Value: {}'.format(type(a_val),a_val))
    print('b_val: Type: {} || Value: {}'.format(type(b_val),b_val))
    print('c_val: Type: {} || Value: {}'.format(type(c_val),c_val))
    
    op_1_val, op_2_val = sess.run([op_1,op_2])
    print('OPERATIONS RESULT: ')
    print('op_1 = {}'.format(op_1_val))
    print('op_2 = {}'.format(op_2_val))
    

