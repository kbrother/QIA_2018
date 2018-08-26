import tensorflow as tf
import pdb
import numpy as np
from tensorflow.contrib import rnn

flags=tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_float('decay', 0.9, 'decay')
flags.DEFINE_integer('epochs', 100, 'number of epoch')

def init_variables(shape):
    return tf.Variable(tf.random_normal (shape, stddev=0.01, dtype=tf.float64))

class many_to_one:
    
    def __init__(self):
       
        self.features = np.load('./data/ita_image_data/activations.npy')  
        self.labels = np.load('./data/ita_image_data/data_labels.npy')
               
        self.features_placeholder = tf.placeholder(self.features.dtype, 
                                              self.features.shape)
        self.labels_placeholder = tf.placeholder(self.labels.dtype, 
                                            self.labels.shape)
        
        assert self.features.shape[0] == self.labels.shape[0]

    def init_iterator(self, sess):        
        sess.run(self.iterator.initializer, feed_dict={self.features_placeholder: self.features,
                                                       self.labels_placeholder: self.labels})

    def load_data(self):
        
        dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, 
                                                      self.labels_placeholder))

        dataset = dataset.shuffle(10000).batch(FLAGS.batch_size)
        self.iterator = dataset.make_initializable_iterator() 
  
    def model(self, vectors, labels):
        
        lstm = rnn.BasicLSTMCell(7)        
        
        inputs = tf.unstack(vectors, 20, 1)
        outputs, _ = rnn.static_rnn(lstm, inputs,
                                   dtype=tf.float64)

        output_w = init_variables([lstm.state_size[1], 7])
        output_b = init_variables([7])

        softmax_input = tf.matmul(outputs[-1], output_w) + output_b          
        softmax = tf.nn.softmax_cross_entropy_with_logits(logits=softmax_input,
                                                          labels=labels)

        return tf.reduce_mean(softmax)


    def train_model(self):
              
        self.load_data()
        vectors, labels = self.iterator.get_next()
        loss = self.model(vectors, labels)
        
        train_op = tf.train.RMSPropOptimizer(FLAGS.learning_rate, 
                                            FLAGS.decay).minimize(loss)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
     
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            for i in range(FLAGS.epochs):
                self.init_iterator(sess)

                while True:
                    try:                    
                        sess.run(train_op)
                        print("success" + "epoch" + str(i))

                    except tf.errors.OutOfRangeError:
                        print("fail")
                        break


m_to_o = many_to_one()
m_to_o.train_model()
