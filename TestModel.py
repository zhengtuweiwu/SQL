#!/usr/bin/env python
# -*- coding:utf-8 -*-
import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import LoadData as ld


if __name__ == '__main__':

    '''# 设置按需使用GPU
    config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    config.operation_timeout_in_ms=-1
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)#最多占gpu资源的70%
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config) # 设置按需使用GPU
    '''
    # SQL脚本长度
    sqlLength = 80
    #RNN隐藏层节点数
    hidden_size=128
    # 测试脚本路径及文件名
    #test_sql_filepath = "test_data\\sql-ex-80-test-lenfilter.txt"
    #test_normal_filepath = "test_data\\normal-ex-80-test-lenfilter.txt"
    #test_filepath = "more_test_data\\test-sql-80.txt"
    #test_filepath = "more_test_data\\test-sql-url-FFDD-80.txt"

    #test_filepath = "test_data\\normal-test-80.txt"
    #test_filepath = "test_data\\normal-ex-test-80.txt"
    #test_filepath = "more_test_data\\test-sql-like-400.txt"
    test_filepath = "more_test_data\\test-sql-like-url-FFDD-80.txt"
    # 测试batch长度
    #test_batch_size = 300
############以下为模型离线加载，测试集测试部分
    # 先加载图和变量
    print("start to test!")
    #重启会话流

    with tf.device('/cpu:0'):
        config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False,device_count={'cpu': 0})
        sess = tf.InteractiveSession(config=config)
        #sess = tf.Session()
        #with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph("model\\GRU\\NO-FFT\\"+str(hidden_size)+"port\\sql-ex-cnn-"+str(sqlLength)+"\\sql-ex-cnn-"+str(sqlLength)+".meta")
        new_saver.restore(sess, tf.train.latest_checkpoint("model\\GRU\\NO-FFT\\"+str(hidden_size)+"port\\sql-ex-cnn-"+str(sqlLength)))
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
        data_batch_test, label_batch_test,count,data_line = ld.get_test_all_txt_single(test_filepath,1,sqlLength)
        data_batch_test = tf.reshape(data_batch_test, [count, sqlLength])
        data_batch_test = tf.cast(data_batch_test, tf.float32) * (1 / 255)
        label_batch_test = tf.reshape(label_batch_test, [count, 2])
        label_batch_test = tf.cast(label_batch_test, tf.float32)
        data_test, label_test = sess.run([data_batch_test, label_batch_test])
        start=time.time()
        accuracy_end, correct_prediction_end,predict_end,out_end = sess.run([accuracy, correct_prediction,predict,out],
                                                            feed_dict={intput: data_test, output: label_test,
                                                                       keep_prob: 1.0})
        end = time.time()
        print("Over more test job in %s s"  % (end-start))
        print("test accuracy %g" % accuracy_end)
        print("test line count %g" % count)
        # print(out_end)
        # print(predict_end)
        iii = 0
        icount = 0
        for isp in correct_prediction_end:
            if isp == True:
                # print(data_line[iii])
                icount = icount + 1
            iii = iii + 1
        print("correct judge line count %g" % icount)

        '''iiii=0
        while iiii<1000:
            # data_batch_test, label_batch_test,count = ld.get_test_all_txt_single(test_filepath, 1,sqlLength)
            data_batch_test, label_batch_test, count = ld.get_test_random_txt_single(test_filepath, 1, 20000, sqlLength)
            count = 20000
            data_batch_test = tf.reshape(data_batch_test, [count, sqlLength])
            data_batch_test = tf.cast(data_batch_test, tf.float32) * (1 / 255)
            label_batch_test = tf.reshape(label_batch_test, [count, 2])
            label_batch_test = tf.cast(label_batch_test, tf.float32)
            data_test, label_test= sess.run([data_batch_test, label_batch_test])
            accuracy_end,correct_prediction_end = sess.run([accuracy,correct_prediction], feed_dict={intput: data_test, output: label_test, keep_prob: 1.0})
            print("test accuracy %g" % accuracy_end)
            print("test line count %g" % count)
            #print(out_end)
            #print(predict_end)
            iii=0
            icount=0
            for isp in correct_prediction_end:
                if isp==True:
                    #print(data_line[iii])
                    icount=icount+1
                iii=iii+1
            print("correct judge line count %g" % icount)
            iiii=iiii+1'''
        sess.close()
