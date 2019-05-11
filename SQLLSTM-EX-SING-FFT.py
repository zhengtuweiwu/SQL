#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import LoadData as ld
from tensorflow.contrib import rnn


if __name__ == '__main__':
    # 设置按需使用GPU
    config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    config.operation_timeout_in_ms=-1
    #config.operation_timeout_in_ms = True
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)#最多占gpu资源的70%
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config) # 设置按需使用GPU

    # SQL脚本长度
    sqlLength = 400
    # 训练脚本路径及文件名
    sql_filepath = "tf_data\\sql-ex-train-400.tf"
    normal_filepath = "tf_data\\normal-ex-train-400.tf"
    # 训练batch长度
    batch_size = 128
    # 迭代次数
    epochs = 230000
    # 测试脚本路径及文件名
    test_sql_filepath = "test_data\\sql-ex-test-400.txt"
    test_normal_filepath = "test_data\\normal-ex-test-400.txt"
    # 测试batch长度
    test_batch_size = 20000

    # 在 1.0 版本以后请使用 ：
    # keep_prob = tf.placeholder(tf.float32, [])
    # batch_size = tf.placeholder(tf.int32, [])

    # 每个时刻的输入特征是1维的，就是每个时刻输入sqlLength个值
    input_size = sqlLength
    # 时序持续长度为1
    timestep_size = 1
    # 每个隐含层的节点数
    hidden_size = 256
    # LSTM/GRU layer 的层数
    layer_num = 3
    ############################获取数据部分
    # 读取文件生成队列
    filename_queue1 = tf.train.string_input_producer([sql_filepath], shuffle=False, num_epochs=None)
    filename_queue2 = tf.train.string_input_producer([normal_filepath], shuffle=False, num_epochs=None)
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
    # 转换数据 sql
    data1 = tf.decode_raw(features1['data'], tf.int8)
    data1 = tf.reshape(data1, [sqlLength])
    data1 = tf.cast(data1, tf.float32) * (1 / 255)
    label1 = tf.cast(features1['label'], tf.int32)
    data_batch1, label_batch1 = tf.train.shuffle_batch([data1, label1], batch_size=int(batch_size / 2), capacity=200,
                                                       min_after_dequeue=100, num_threads=4)
    # data_batch1, label_batch1 = tf.train.batch([data1, label1], batch_size=int(batch_size / 2),num_threads=4,allow_smaller_final_batch=False,capacity=500)
    # 转换数据 normal
    data2 = tf.decode_raw(features2['data'], tf.int8)
    data2 = tf.reshape(data2, [sqlLength])
    data2 = tf.cast(data2, tf.float32) * (1 / 255)
    label2 = tf.cast(features2['label'], tf.int32)
    data_batch2, label_batch2 = tf.train.shuffle_batch([data2, label2], batch_size=int(batch_size / 2), capacity=200,
                                                       min_after_dequeue=100, num_threads=4)
    # data_batch2, label_batch2 = tf.train.batch([data2, label2], batch_size=int(batch_size / 2),num_threads=4,allow_smaller_final_batch=False,capacity=500)
    ####################################测试数据获取部分
    data_batch_test, label_batch_test = ld.get_test_random_txt(test_sql_filepath, test_normal_filepath, test_batch_size,
                                                               sqlLength)
    # data_batch_test = tf.reshape(data_batch_test, [test_batch_size, sqlLength])
    data_batch_test = tf.cast(data_batch_test, tf.float32) * (1 / 255)
    # label_batch_test = tf.reshape(label_batch_test, [test_batch_size, 2])
    label_batch_test = tf.cast(label_batch_test, tf.float32)



    ############################模型定义部分
    X_ = tf.placeholder(tf.float32, [None, sqlLength], name="input")
    y_ = tf.placeholder(tf.float32, [None, 2], name="output")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    # 下面几个步骤是实现 RNN / LSTM 的关键
    ####################################################################
    # **步骤1：RNN 的输入shape = (batch_size, timestep_size, input_size)
    # 把X_转为卷积所需要的形式 加入FFT快速转换 将原矩阵跟转换后的矩阵合并
    X_ = tf.cast(X_, tf.complex64)
    X_FFT = tf.fft(X_)
    X_INPUT = tf.concat([X_, X_FFT], 1)
    X_INPUT = tf.cast(X_INPUT, tf.float32)
    X_INPUT = tf.reshape(X_INPUT, [-1, 1, 2 * sqlLength], name="fft_input")

    # **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
    lstm_cell_b = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)

    # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
    lstm_cell_b = rnn.DropoutWrapper(cell=lstm_cell_b, input_keep_prob=1.0, output_keep_prob=keep_prob)
    # **步骤4：初始化第一层 并且单独输出结果，并且使其运行起来
    #init_state = lstm_cell_b.zero_state(batch_size, dtype=tf.float32)#如果加载此处参数，容易固化网络输入状态，导致智能输入batch数量参数检测,坑点！！！
    outputs, state = tf.nn.dynamic_rnn(lstm_cell_b, inputs=X_INPUT, initial_state=None, time_major=False,dtype=tf.float32)#如果加载此处参数，容易固化网络输入状态，导致智能输入batch数量参数检测,坑点！！！
    #第一层输出结果
    h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]
    if layer_num >= 2:
        #print(h_state)
        s_X=tf.reshape(h_state, [-1,1,hidden_size])
        #print(s_X)
        # **步骤5：定义顶层LSTM_cell，可以配置多层
        lstm_cell_u = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
        # **步骤6：添加 dropout layer, 一般只设置 output_keep_prob
        lstm_cell_u = rnn.DropoutWrapper(cell=lstm_cell_u, input_keep_prob=1.0, output_keep_prob=keep_prob)

        # **步骤7：调用 MultiRNNCell 来实现顶层的多层 LSTM
        mlstm_cell = rnn.MultiRNNCell(cells=[lstm_cell_u]*(layer_num-1), state_is_tuple=True)

        # **步骤8：用全零来初始化顶部多层的state
        #init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)#如果加载此处参数，容易固化网络输入状态，导致智能输入batch数量参数检测,坑点！！！

        # **步骤9：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
        # ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
        # ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
        # ** state.shape = [layer_num, 2, batch_size, hidden_size],
        # ** 或者，可以取 h_state = state[-1][1] 作为最后输出
        # ** 最后输出维度是 [batch_size, hidden_size]
        outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=s_X, initial_state=None, time_major=False,dtype=tf.float32)#如果加载此处参数，容易固化网络输入状态，导致智能输入batch数量参数检测,坑点！！！
        h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]

    # *************** 为了更好的理解 LSTM 工作原理，我们把上面 步骤6 中的函数自己来实现 ***************
    # 通过查看文档你会发现， RNNCell 都提供了一个 __call__()函数（见最后附），我们可以用它来展开实现LSTM按时间步迭代。
    # **步骤9：方法二，按时间步展开计算
    '''outputs = list()
    state = init_state
    with tf.variable_scope('RNN'):
        for timestep in range(timestep_size):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            # 这里的state保存了每一层 LSTM 的状态
            (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
            outputs.append(cell_output)
    h_state = outputs[-1]'''

    # 上面 LSTM 部分的输出会是一个 [hidden_size] 的tensor，我们要分类的话，还需要接一个 softmax 层
    # 首先定义 softmax 的连接权重矩阵和偏置
    # out_W = tf.placeholder(tf.float32, [hidden_size, class_num], name='out_Weights')
    # out_bias = tf.placeholder(tf.float32, [class_num], name='out_bias')
    # 开始训练和测试
    W = tf.Variable(tf.truncated_normal([hidden_size, 2], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1, shape=[2]), dtype=tf.float32)
    y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias,name="out")
    y_predict = tf.round(y_pre, name="predict")
    # 损失和评估函数
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_pre, 1e-10, 1.0)), name="loss")
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_, 1),name="correct_prediction")
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),name="accuracy")
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("correct_prediction", correct_prediction)
    tf.add_to_collection("loss", cross_entropy)
    tf.add_to_collection("predict", y_predict)
    tf.add_to_collection("out", y_pre)
    ###########初始化全局参数
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    # 获得协调对象
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  ####没有此处协调全局变量后面读写进程在session关闭后会报错
    for i in range(epochs):
        # print("step %d" % (i))
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
        # print(data_convert)
        # print(label_convert)
        if i % 1000 == 0:
            loss, train_accuracy = sess.run([cross_entropy, accuracy],
                                            feed_dict={X_: data_convert, y_: label_convert, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            print("step %d, training loss %g" % (i, loss / batch_size))
        train_step.run(feed_dict={X_: data_convert, y_: label_convert, keep_prob: 0.5})
    #########模型保存
    saver = tf.train.Saver()
    saver.save(sess, "model\\LSTM\\SINGAL-FFT\\3layers\\sql-ex-cnn-" + str(sqlLength) + "\\sql-ex-cnn-" + str(sqlLength))
    ###########以下为现场测试
    data_test, label_test = sess.run([data_batch_test, label_batch_test])
    loss_end, accuracy_end = sess.run([cross_entropy, accuracy],
                                      feed_dict={X_: data_test, y_: label_test, keep_prob: 1.0})
    print("test accuracy %g" % accuracy_end)
    # print("test loss %g" % (loss_end / batch_size))
    # 关闭会话流
    coord.request_stop()  # 请求线程结束
    coord.join(threads)  # 等待线程结束 没有此处读写进程会出现异常报错
    sess.close()
    ############以下为模型离线加载，测试集测试部分
    # 先加载图和变量
    print("start to test!")
    # 重启会话流
    with tf.Session() as sess_test:
        new_saver = tf.train.import_meta_graph(
            "model\\LSTM\\SINGAL-FFT\\3layers\\sql-ex-cnn-" + str(sqlLength) + "\\sql-ex-cnn-" + str(sqlLength) + ".meta")
        new_saver.restore(sess_test, tf.train.latest_checkpoint("model\\LSTM\\SINGAL-FFT\\3layers\\sql-ex-cnn-" + str(sqlLength)))
        accuracy = tf.get_collection("accuracy")[0]
        predict = tf.get_collection("predict")[0]
        out = tf.get_collection("out")[0]
        loss = tf.get_collection("loss")[0]
        correct_prediction = tf.get_collection("correct_prediction")[0]
        graph = tf.get_default_graph()
        # placeholders 操作用get_operation_by_name
        intput = graph.get_operation_by_name("input").outputs[0]  # 输入数据 此处
        output = graph.get_operation_by_name("output").outputs[0]  # 对照输入标签
        keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]  # 随机保留节点比例

        data_batch_test, label_batch_test = ld.get_test_random_txt(test_sql_filepath, test_normal_filepath,
                                                                   test_batch_size, sqlLength)
        data_batch_test = tf.reshape(data_batch_test, [test_batch_size, sqlLength])
        data_batch_test = tf.cast(data_batch_test, tf.float32) * (1 / 255)
        label_batch_test = tf.reshape(label_batch_test, [test_batch_size, 2])
        label_batch_test = tf.cast(label_batch_test, tf.float32)

        data_test, label_test = sess_test.run([data_batch_test, label_batch_test])
        accuracy_end, predict_end, out_end, loss_end = sess_test.run([accuracy, predict, out, loss],
                                                                     feed_dict={intput: data_test, output: label_test,
                                                                                keep_prob: 1.0})
        print("test accuracy %g" % accuracy_end)
        # print("test loss %g" % (loss_end/batch_size))
        # print(out_end)
        # print(predict_end)
        # print(correct_prediction_end)