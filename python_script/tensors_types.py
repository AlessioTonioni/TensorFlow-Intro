
# coding: utf-8

# ## Variables, constant and placeholder
# We are going to create a variable, a constant and a placeholder containing a 2X2 identyty matrix.

# In[ ]:


import tensorflow as tf
import numpy as np

a_const = tf.constant([[1,0],[0,1]], name="a_const")
a_variable = tf.Variable([[1,0],[0,1]],trainable=False, name="a_var")
a_placeholder = tf.placeholder(tf.int64,shape=(2,2), name="a_placeholder")
with tf.variable_scope('my_first_scope') as scope:
    a_scoped_variable = tf.Variable([[1,0],[0,1]],trainable=False, name="a_var")

print(a_const)
print(a_variable)
print(a_placeholder)
print(a_scoped_variable)


# Let's now try to evaluate them inside a session:
# ### Constant

# In[ ]:


with tf.Session() as sess:
    a_value = sess.run(a_const)
    print(a_value)


# ### Variable

# In[ ]:


with tf.Session() as sess:
    a_value = sess.run(a_variable)


# Initialization is missing!!!! 
# Proper way:

# In[ ]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a_value = sess.run(a_variable)
    print(a_value)
    a_value = sess.run(a_scoped_variable)
    print(a_value)


# ### Placeholder

# In[ ]:


with tf.Session() as sess:
    a_value = sess.run(a_placeholder, feed_dict={a_placeholder:[[1,0],[0,1]]})
    print(a_value)

