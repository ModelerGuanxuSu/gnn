import tensorflow as tf
from tensorflow.python.keras.initializers import Identity, glorot_uniform, Zeros
from tensorflow.python.keras.layers import Dropout, Input, Layer, Embedding, Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
import numpy as np
import random
import datetime
from sklearn import datasets
import scipy.sparse as sp
import copy

def layer_A(A,X,n_nodes,drop_out_rate,input_dim,activation=None):
    
    W = tf.Variable(tf.random_normal([input_dim,n_nodes]),dtype=tf.float32)
#     tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(lamda)(W))
    b = tf.Variable(tf.zeros([n_nodes]),dtype=tf.float32)
    
    drop_out_X = tf.nn.dropout(X, rate = drop_out_rate)
    AX = tf.sparse.sparse_dense_matmul(A,drop_out_X)
    if activation is not None:
        output = activation(tf.matmul(AX,W)+b)
    else:
        output = tf.matmul(AX,W)+b
    return(output,W)


class GNN(object):
    '''
Parameters
----------
layers : list of numbers of nodes in each hidden layer

batch_size : int, size of each batch, default : 256

epoch : int, number of epoch, default : 10

activation : tensorflow activation, default: tf.nn.relu

dropout_rate : float, percentage of weights in each layer 
    to drop during training, default : 0.5

l2_reg : float, L2 norm for all the weights, default : 1e-5

learning_rate : float, learning rate, default : 0.01

    '''
    def __init__(self,layers=[16,16],batch_size=256,epoch = 10,activation=tf.nn.relu,\
                 dropout_rate=0.5, l2_reg=1e-5,learning_rate=0.01):
        self.layers = layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        
        
    def fit(self,X,Y,train_mask,A=None,track=True):
        '''
            Inputs:
                X: ndarray , shape of (n,p)

                Y: ndarray , one-hot encoded, shape of (n,#class)

                train_mask: ndarray, float, shape of (n,)
                    Only those observation k with train_mask[k]>0 will be trained.
                    The value of each element can be regarded as the weight of each
                    observation in loss function.

                A: scipy.sparse.coo.coo_matrix, shape of (n,n)
                    If A is None, identity matrix will be adopted,
                    in such case, the network is equivalent to a normal NN. 
                    Default : None

                track: bool, whether to print loss track, default : True
        '''

        tf.reset_default_graph()

        if A is None:
            row_ = np.array(range(X.shape[0]))
            column_ = np.array(range(X.shape[0]))
            data_ = np.ones_like(column_)
            A = sp.coo_matrix((data_, (row_, column_)),
                    shape=(X.shape[0], X.shape[0]), dtype=np.float32)
        
        indices = np.array([i for i in zip(A.row,A.col)],np.int)
        # A = tf.compat.v1.SparseTensorValue(indices, A.data, A.shape)
        
        ##定义网络结构
        x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, X.shape[1]], name='x')
        y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, Y.shape[1]], name='y')
        mask = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='mask')
        # a = tf.compat.v1.sparse_placeholder(dtype=tf.float32, name='adj')
        a_indices = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None,None], name='a_indices')
        a_data = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,], name='a_data')
        a_shape = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None,], name='a_shape')
        a = tf.compat.v1.SparseTensorValue(a_indices, a_data, a_shape)
        rate = tf.compat.v1.placeholder('float',name='rate')
        
        W_list = []
        for l in range(len(self.layers)):
            if l==0:
                h = x
                h,W = layer_A(a,h,self.layers[l],rate,X.shape[1],self.activation)
                W_list.append(W)
                continue
            h,W = layer_A(a,h,self.layers[l],rate,self.layers[l-1],self.activation)
            W_list.append(W)
        ##预测值
        y_logit,W = layer_A(a,h,Y.shape[1],rate,self.layers[-1])
        W_list.append(W)
#         tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(self.l2_reg)(W))
        y_proba = tf.nn.softmax(y_logit,name = 'y_proba')
        ##损失函数
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y,y_logit,weights=mask))

        ##正则罚项
        for w in W_list:
            loss += tf.reduce_mean(tf.square(w))

        ##train step
        global_step = tf.Variable(0)
        train_step = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=global_step)
        
        ##参数训练
        sess = tf.compat.v1.Session()

        init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        sess.run(init_op)
        Loss = []
        step_ = 0
        index_train = [i for i in range(len(X)) if train_mask[i]>0]
        for _ in range(self.epoch):
            index_all = copy.deepcopy(index_train)
            while len(index_all)>0:
                if len(index_all)>=self.batch_size:
                    index_batch = random.sample(index_all,self.batch_size)
                    index_all = list(set(index_all)-set(index_batch))
                else:
                    index_batch = index_all
                    index_all = []
                    
                step_ += 1
                batch_mask = np.zeros((len(Y),))
                batch_mask[index_batch] = 1

            #     sess.run(train_step, feed_dict={x: X, y: Y, a:A, mask:batch_mask, rate:self.dropout_rate})
            #     Loss.append(sess.run(loss, feed_dict={x: X, y: Y, a:A, mask:batch_mask, rate:0.0}))
            # loss_all = sess.run(loss, feed_dict={x: X, y: Y, a:A, mask:train_mask, rate:0.0})
                sess.run(train_step, feed_dict={x: X, y: Y, 
                    a_shape:A.shape,a_indices:indices,a_data:A.data,
                    mask:batch_mask, rate:self.dropout_rate})
                Loss.append(sess.run(loss, feed_dict={x: X, y: Y, 
                    a_shape:A.shape,a_indices:indices,a_data:A.data,
                    mask:batch_mask, rate:0.0}))
            loss_all = sess.run(loss, feed_dict={x: X, y: Y, 
                a_shape:A.shape,a_indices:indices,a_data:A.data,
                mask:train_mask, rate:0.0})

            time_ = datetime.datetime.now().strftime('%Y-%D %H:%M:%S')
            if track:
                print(time_ + '\tepoch ' + str(_) + '\t' + str(loss_all))
            
        self.__sess = sess
        self.__x = x
        self.__y = y
        self.__a_shape = a_shape
        self.__a_data = a_data
        self.__a_indices = a_indices
        self.__y_proba = y_proba
        self.__rate = rate
        self.__mask = mask
        # self.__is_training = is_training
        
    def predict_proba(self,X,A=None):
        '''
            Inputs:
                X: ndarray , shape of (n,p)
                A: scipy.sparse.coo.coo_matrix, shape of (n,n)
                    If A is None, identity matrix will be adopted,
                    in such case, the network is equivalent to a normal NN. 
            Outputs:
                y_pred: ndarray , shape of (n,#class)
        '''
        if A is None:
            row_ = np.array(range(X.shape[0]))
            column_ = np.array(range(X.shape[0]))
            data_ = np.ones_like(column_)
            A = sp.coo_matrix((data_, (row_, column_)),
                    shape=(X.shape[0], X.shape[0]), dtype=np.float32)

        indices = np.array([i for i in zip(A.row,A.col)],np.int)
        # A = tf.compat.v1.SparseTensorValue(indices, A.data, A.shape)
        y_proba = self.__sess.run(self.__y_proba,\
            feed_dict={self.__x: X, 
            self.__a_shape:A.shape, self.__a_data:A.data, self.__a_indices:indices,
            self.__rate:0.0})
        return(y_proba)
    
    def predict(self,X,A=None):
        '''
            Inputs:
                X: ndarray , shape of (n,p)
                A: scipy.sparse.coo.coo_matrix, shape of (n,n)
                    If A is None, identity matrix will be adopted,
                    in such case, the network is equivalent to a normal NN. 
            Outputs:
                y_pred: ndarray , shape of (n,)
        '''
        if A is None:
            row_ = np.array(range(X.shape[0]))
            column_ = np.array(range(X.shape[0]))
            data_ = np.ones_like(column_)
            A = sp.coo_matrix((data_, (row_, column_)),
                    shape=(X.shape[0], X.shape[0]), dtype=np.float32)

        indices = np.array([i for i in zip(A.row,A.col)],np.int)
        A = tf.compat.v1.SparseTensorValue(indices, A.data, A.shape)

        y_proba = self.predict_proba(X,A)
        bool_ = y_proba==np.transpose(np.ones_like(y_proba.T)*np.max(y_proba,axis=1))
        index_ = np.ones_like(y_proba)*range(y_proba.shape[1])
        y_pred = index_[bool_]
        return(y_pred)

    def save_parameters(self,save_dict):
        variables_to_resotre  = tf.contrib.framework.get_variables_to_restore()
        saver = tf.compat.v1.train.Saver(variables_to_resotre)
        save_path = saver.save(self.__sess,save_dict)
        print(save_path)

    def load_parameters(self,load_dict):
        saver = tf.compat.v1.train.import_meta_graph(load_dict+'.meta')
        sess = tf.compat.v1.Session()
        saver.restore(sess, load_dict)
        graph = tf.compat.v1.get_default_graph()
        self.__sess = sess
        self.__x = graph.get_tensor_by_name("x:0")
        self.__y = graph.get_tensor_by_name("y:0")
        self.__y_proba = graph.get_tensor_by_name("y_proba:0")
        self.__rate = graph.get_tensor_by_name("rate:0")
        self.__mask = graph.get_tensor_by_name("mask:0")
        self.__a_shape = graph.get_tensor_by_name("a_shape:0")
        self.__a_data = graph.get_tensor_by_name("a_data:0")
        self.__a_indices = graph.get_tensor_by_name("a_indices:0")

##由一个稀疏矩阵得到GCN定义的A矩阵
def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm

def get_A_GCN(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj.tocoo()


####获取数据

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def convert_symmetric(X, sparse=True):
    if sparse:
        X += X.T - sp.diags(X.diagonal())
    else:
        X += X.T - np.diag(X.diagonal())
    return X


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def get_splits(y,):
    idx_list = np.arange(len(y))
    # train_val, idx_test = train_test_split(idx_list, test_size=0.2, random_state=1024)  # 1000
    # idx_train, idx_val = train_test_split(train_val, test_size=0.2, random_state=1024)  # 500

    idx_train = []
    label_count = {}
    for i, label in enumerate(y):
        label = np.argmax(label)
        if label_count.get(label, 0) < 20:
            idx_train.append(i)
            label_count[label] = label_count.get(label, 0) + 1

    idx_val_test = list(set(idx_list) - set(idx_train))
    idx_val = idx_val_test[0:500]
    idx_test = idx_val_test[500:1500]


    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    val_mask = sample_mask(idx_val, y.shape[0])
    test_mask = sample_mask(idx_test, y.shape[0])

    return y_train, y_val, y_test,train_mask, val_mask, test_mask


def load_data_v1(dataset="cora", path="./data/cora/",):

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    onehot_labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = convert_symmetric(adj, )

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_splits(onehot_labels)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask