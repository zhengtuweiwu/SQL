#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import LoadData as ld

# 权值初始化
def weight_variable(shape):
    # 用正态分布来初始化权值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 本例中用relu激活函数，所以用一个很小的正偏置较好
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
# 定义卷积层
def conv2d(x, W):
    # 默认 strides[0]=strides[3]=1, strides[1]为x方向步长，strides[2]为y方向步长
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
# pooling 层
def max_pool_2x2(x):
    #由于只有一维，因此只在一个方向进行padding
    return tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')

if __name__ == '__main__':
    ##########
    # tf.ConfigProto()的参数如下：
    # log_device_placement=True : 是否打印设备分配日志
    # allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备
    # tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
    # 在构造tf.Session()时可通过tf.GPUOptions作为可选配置参数的一部分来显示地指定需要分配的显存比例。
    # per_process_gpu_memory_fraction指定了每个GPU进程中使用显存的上限，但它只能均匀地作用于所有GPU，无法对不同GPU设置不同的上限。
    ##########
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)#配置tf显存最大使用空间
    # sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False))
    # 设置按需使用GPU
    #sess = tf.Session()
    config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    config.operation_timeout_in_ms=-1
    #config.operation_timeout_in_ms = True
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)#最多占gpu资源的70%
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config) # 设置按需使用GPU
    #卷积层窗口大小
    stride=5
    #SQL脚本长度
    sqlLength=320
    #训练脚本路径及文件名
    sql_filepath="tf_data\\sql-ex-train-320.tf"
    normal_filepath = "tf_data\\normal-ex-train-320.tf"
    #训练batch长度
    batch_size=128
    #迭代次数
    epochs=160000
    # 测试脚本路径及文件名
    test_sql_filepath = "test_data\\sql-ex-test-320.txt"
    test_normal_filepath = "test_data\\normal-ex-test-320.txt"
    # 测试batch长度
    test_batch_size = 20000
    ############################获取数据部分
    # 读取文件生成队列
    filename_queue1 = tf.train.string_input_producer([sql_filepath], shuffle=False,num_epochs=None)
    filename_queue2 = tf.train.string_input_producer([normal_filepath],shuffle=False, num_epochs=None)
    # 生成reader流
    reader1 = tf.TFRecordReader()
    reader2 = tf.TFRecordReader()
    _, serialized_example1 = reader1.read(filename_queue1)
    _, serialized_example2 = reader2.read(filename_queue2)
    # get feature from serialized example
    features1 = tf.parse_single_example(serialized_example1,
                                       features={
                                           'data': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64)
                                       }
                                       )
    features2 = tf.parse_single_example(serialized_example2,
                                       features={
                                           'data': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64)
                                       }
                                       )
    #转换数据 sql
    data1 = tf.decode_raw(features1['data'], tf.int8)
    data1 = tf.reshape(data1, [sqlLength])
    data1 = tf.cast(data1, tf.float32) * (1 / 255)
    label1 = tf.cast(features1['label'], tf.int32)
    data_batch1, label_batch1 = tf.train.shuffle_batch([data1, label1], batch_size=int(batch_size/2), capacity=200,
                                                    min_after_dequeue=100, num_threads=4)
    #data_batch1, label_batch1 = tf.train.batch([data1, label1], batch_size=int(batch_size / 2),num_threads=4,allow_smaller_final_batch=False,capacity=500)
    # 转换数据 normal
    data2 = tf.decode_raw(features2['data'], tf.int8)
    data2 = tf.reshape(data2, [sqlLength])
    data2 = tf.cast(data2, tf.float32) * (1 / 255)
    label2 = tf.cast(features2['label'], tf.int32)
    data_batch2, label_batch2 = tf.train.shuffle_batch([data2, label2], batch_size=int(batch_size / 2), capacity=200,
                                                       min_after_dequeue=100, num_threads=4)
    #data_batch2, label_batch2 = tf.train.batch([data2, label2], batch_size=int(batch_size / 2),num_threads=4,allow_smaller_final_batch=False,capacity=500)
    ####################################测试数据获取部分
    data_batch_test, label_batch_test = ld.get_test_random_txt(test_sql_filepath, test_normal_filepath, test_batch_size,sqlLength)
    # data_batch_test = tf.reshape(data_batch_test, [test_batch_size, sqlLength])
    data_batch_test = tf.cast(data_batch_test, tf.float32) * (1 / 255)
    # label_batch_test = tf.reshape(label_batch_test, [test_batch_size, 2])
    label_batch_test = tf.cast(label_batch_test, tf.float32)
    ############################模型定义部分
    X_ = tf.placeholder(tf.float32, [None, sqlLength],name="input")
    y_ = tf.placeholder(tf.float32, [None, 2],name="output")
    # 把X_转为卷积所需要的形式
    X = tf.reshape(X_, [-1, 1, sqlLength, 1])
    #print(X)
    # 第一层卷积：1×5×1卷积核32个 [1，5，1，32],h_conv1.shape=[-1, 1, sqlLength, 32]
    W_conv1 = weight_variable([1, stride, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
    # 第一个pooling 层[-1, 1, sqlLength, 32]->[-1, 1, sqlLength/2, 32]
    h_pool1 = max_pool_2x2(h_conv1)
    # 第二层卷积：1×5×32卷积核64个 [1，5，32，64],h_conv2.shape=[-1, 1, sqlLength/2, 64]
    W_conv2 = weight_variable([1, stride, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # 第二个pooling 层,[-1, 1, sqlLength/2, 64]->[-1, 1, sqlLength/4, 64]
    h_pool2 = max_pool_2x2(h_conv2)
    # flatten层，[-1, 1, sqlLength/4, 64]->[-1, 7*(sqlLength/4)*64],即每个样本得到一个1 * (sqlLength/4) * 64维的样本
    h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * (int(sqlLength/4)) * 64])
    # fc1
    #W_fc1 = weight_variable([7 * 7 * 64, 1024])
    W_fc1 = weight_variable([1 * (int(sqlLength/4)) * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
    keep_prob = tf.placeholder(tf.float32,name="keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # 输出层
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name="out")
    y_predict=tf.round(y_conv,name="predict")
    #此种方案容易出现NAN值
    #cross_entropy = -tf.reduce_mean(y_ * tf.log(y_conv)) #tf.reduce_sum/reduce_mean   是取batch个样本集中损失的和还是平均值有待商榷
    #解决方案一 该方案可行
    #tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。
    #小于min的让它等于min，大于max的元素的值等于max
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)),name="loss")
    #解决方案二 原理同一
    #cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv + 1e-10))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1),name="correct_prediction")
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),name="accuracy")
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("correct_prediction", correct_prediction)
    tf.add_to_collection("loss", cross_entropy)
    tf.add_to_collection("predict", y_predict)
    tf.add_to_collection("out", y_conv)
    ###########初始化全局参数
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    # 获得协调对象
    coord = tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)####没有此处协调全局变量后面读写进程在session关闭后会报错
    for i in range(epochs):
        #print("step %d" % (i))
        ######获取数据转换部分
        data_val1, label_val1 = sess.run([data_batch1, label_batch1])
        data_val2, label_val2 = sess.run([data_batch2, label_batch2])
        data_convert = []
        label_convert = []
        ii = 0
        while ii < int(batch_size / 2):
            if label_val1[ii] == 0:
                label_convert.append([0, 1])
            else:
                label_convert.append([1, 0])
            data_convert.append(data_val1[ii])
            if label_val2[ii] == 0:
                label_convert.append([0, 1])
            else:
                label_convert.append([1, 0])
            data_convert.append(data_val2[ii])
            ii = ii + 1
        data_convert = np.asarray(data_convert)
        label_convert = np.asarray(label_convert)
        # np.arange 生成随机序列
        shuffle_indices = np.random.permutation(np.arange(batch_size))
        data_convert = data_convert[shuffle_indices]
        label_convert = label_convert[shuffle_indices]
        #print(data_convert)
        #print(label_convert)
        if i % 1000 == 0:
            loss, train_accuracy = sess.run([cross_entropy, accuracy],feed_dict={X_: data_convert, y_: label_convert, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            print("step %d, training loss %g" % (i, loss/batch_size))
        train_step.run(feed_dict={X_:data_convert, y_: label_convert, keep_prob: 0.5})
    #########模型保存
    saver = tf.train.Saver()
    saver.save(sess, "model\\sql-ex-cnn-" + str(sqlLength) + "\\sql-ex-cnn-" + str(sqlLength))
    ###########以下为模型现场测试
    '''
    data_val1, label_val1 = sess.run([data_batch1, label_batch1])
    data_val2, label_val2 = sess.run([data_batch2, label_batch2])
    data_convert = []
    label_convert = []
    ii = 0
    while ii < int(batch_size / 2):
        if label_val1[ii] == 0:
            label_convert.append([0, 1])
        else:
            label_convert.append([1, 0])
        data_convert.append(data_val1[ii])
        if label_val2[ii] == 0:
            label_convert.append([0, 1])
        else:
            label_convert.append([1, 0])
        data_convert.append(data_val2[ii])
        ii = ii + 1
    data_convert = np.asarray(data_convert)
    label_convert = np.asarray(label_convert)
    # np.arange 生成随机序列
    shuffle_indices = np.random.permutation(np.arange(batch_size))
    data_convert = data_convert[shuffle_indices]
    label_convert = label_convert[shuffle_indices]
    loss, train_accuracy = sess.run([cross_entropy, accuracy],
                                    feed_dict={X_: data_convert, y_: label_convert, keep_prob: 1.0})
    print("test accuracy %g" % train_accuracy)
    print("test loss %g" % (loss / batch_size))
    '''
    data_test, label_test = sess.run([data_batch_test, label_batch_test])
    loss_end , accuracy_end= sess.run([cross_entropy,accuracy], feed_dict={X_:data_test, y_: label_test, keep_prob: 1.0})
    print("test accuracy %g" % accuracy_end)
    #print("test loss %g" % (loss_end / batch_size))
    # 关闭会话流
    coord.request_stop()  # 请求线程结束
    coord.join(threads)  # 等待线程结束 没有此处读写进程会出现异常报错
    sess.close()
    ############以下为模型离线加载，测试集测试部分
    # 先加载图和变量
    print("start to test!")
    #重启会话流
    with tf.Session() as sess_test:
        new_saver = tf.train.import_meta_graph("model\\sql-ex-cnn-"+str(sqlLength)+"\\sql-ex-cnn-"+str(sqlLength)+".meta")
        new_saver.restore(sess_test, tf.train.latest_checkpoint("model\\sql-ex-cnn-"+str(sqlLength)))
        accuracy = tf.get_collection("accuracy")[0]
        predict = tf.get_collection("predict")[0]
        out = tf.get_collection("out")[0]
        loss = tf.get_collection("loss")[0]
        correct_prediction = tf.get_collection("correct_prediction")[0]
        loss = tf.get_collection("loss")[0]
        graph = tf.get_default_graph()
        # placeholders 操作用get_operation_by_name
        intput = graph.get_operation_by_name("input").outputs[0]  # 输入数据 此处
        output = graph.get_operation_by_name("output").outputs[0]  # 对照输入标签
        keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]  # 随机保留节点比例
        '''
        accuracy = graph.get_tensor_by_name("accuracy:0")#准确度
        predict = graph.get_tensor_by_name("predict:0")#转化后的0 1 结果标签
        out = graph.get_tensor_by_name("out:0")#预测标签
        loss = graph.get_tensor_by_name("loss:0")  # 预测标签
        correct_prediction = graph.get_tensor_by_name("correct_prediction:0")
        '''
        data_batch_test, label_batch_test = ld.get_test_random_txt(test_sql_filepath, test_normal_filepath, test_batch_size,sqlLength)
        data_batch_test = tf.reshape(data_batch_test, [test_batch_size, sqlLength])
        data_batch_test = tf.cast(data_batch_test, tf.float32) * (1 / 255)
        label_batch_test = tf.reshape(label_batch_test, [test_batch_size, 2])
        label_batch_test = tf.cast(label_batch_test, tf.float32)

        data_test, label_test= sess_test.run([data_batch_test, label_batch_test])
        accuracy_end,predict_end,out_end,loss_end = sess_test.run([accuracy,predict,out,loss], feed_dict={intput: data_test, output: label_test, keep_prob: 1.0})
        print("test accuracy %g" % accuracy_end)
        #print("test loss %g" % (loss_end/batch_size))
    #print(out_end)
    #print(predict_end)
    #print(correct_prediction_end)


