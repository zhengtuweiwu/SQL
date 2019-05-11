#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import random
import numpy as np
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
# 创建单一record格式文件
def CreatSingleTFRecordFile(black,white,outpath):
    writer = tf.python_io.TFRecordWriter(outpath)
    br = open(black,"rb")
    wr = open(white,"rb")
    lines = []
    labels = []
    #print((chr(0)+chr(1)).encode("utf8", "ignore"))
    #print((chr(1) + chr(0)).encode("utf8", "ignore"))
    for cur_line_number, line in enumerate(br):
        lines.append(line)
        labels.append(0)
    br.close()
        #other标签为1 [1,0]SQL注入（黑）样本标签为0 [0,1]
    for cur_line_number, line in enumerate(wr):
        lines.append(line)
        labels.append(1)
    wr.close()
    lines = np.asarray(lines)
    labels = np.asarray(labels)
    length = lines.shape[0]
    print(length)
    # np.arange 生成随机序列
    shuffle_indices = np.random.permutation(np.arange(length))
    # print(shuffle_indices)
    lines = lines[shuffle_indices]
    labels = labels[shuffle_indices]
    ii = 0
    print("write")
    while (ii < length):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'data': _bytes_feature(lines[ii]),
                    'label': _int64_feature(labels[ii])
                }
            )
        )
        writer.write(example.SerializeToString())
        ii = ii + 1
    writer.close()

    return 0
# 转换成record格式文件
def ConverTFRecordFile(inputpath,outpath,label,length):
    writer = tf.python_io.TFRecordWriter(outpath)
    br = open(inputpath,"rb")
    lines = []
    labels = []
    #print((chr(0)+chr(1)).encode("utf8", "ignore"))
    #print((chr(1) + chr(0)).encode("utf8", "ignore"))
    for cur_line_number, line in enumerate(br):
        #print(len(line))
        if len(line)==length:
            lines.append(line)
            labels.append(label)
    br.close()
        #other标签为1 [1,0]SQL注入（黑）样本标签为0 [0,1]
    lines = np.asarray(lines)
    labels = np.asarray(labels)
    leng = lines.shape[0] 
    # np.arange 生成随机序列
    shuffle_indices = np.random.permutation(np.arange(leng))
    # print(shuffle_indices)
    lines = lines[shuffle_indices]
    labels = labels[shuffle_indices]
    ii = 0
    print(inputpath+"write")
    print(leng)
    while (ii < leng):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'data': _bytes_feature(lines[ii]),
                    'label': _int64_feature(labels[ii])
                }
            )
        )
        writer.write(example.SerializeToString())
        ii = ii + 1
    writer.close()

    return 0
# 创建单一record格式文件
def CreatSingleTFRecordFile(black,white,outpath):
    writer = tf.python_io.TFRecordWriter(outpath)
    br = open(black,"rb")
    wr = open(white,"rb")
    lines = []
    labels = []
    #print((chr(0)+chr(1)).encode("utf8", "ignore"))
    #print((chr(1) + chr(0)).encode("utf8", "ignore"))
    for cur_line_number, line in enumerate(br):
        lines.append(line)
        labels.append(0)
    br.close()
        #other标签为1 [1,0]SQL注入（黑）样本标签为0 [0,1]
    for cur_line_number, line in enumerate(wr):
        lines.append(line)
        labels.append(1)
    wr.close()
    lines = np.asarray(lines)
    labels = np.asarray(labels)
    length = lines.shape[0]
    print(length)
    # np.arange 生成随机序列
    shuffle_indices = np.random.permutation(np.arange(length))
    # print(shuffle_indices)
    lines = lines[shuffle_indices]
    labels = labels[shuffle_indices]
    ii = 0
    print("write")
    while (ii < length):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'data': _bytes_feature(lines[ii]),
                    'label': _int64_feature(labels[ii])
                }
            )
        )
        writer.write(example.SerializeToString())
        ii = ii + 1
    writer.close()

    return 0
# 创建区分训练集测试集record格式文件，加入比例因子
def CreatTFRecordFile(black,white,outpath,scale):
    writer1 = tf.python_io.TFRecordWriter(outpath+"-train.tf")
    writer2 = tf.python_io.TFRecordWriter(outpath +"-test.tf")
    br = open(black,"rb")
    wr = open(white,"rb")
    lines=[]
    labels=[]
    #lines1=br.readlines()
    #random.shuffle(lines1)
    for cur_line_number, line in enumerate(br):
        lines.append(line)
        labels.append(0)
    br.close()
        #other标签为1 [1,0] SQL注入（黑）样本标签为0  [0,1]

    #lines2 = wr.readlines()
    #random.shuffle(lines2)
    for cur_line_number, line in enumerate(wr):
        lines.append(line)
        labels.append(1)
    wr.close()
    lines=np.asarray(lines)
    labels=np.asarray(labels)
    length = lines.shape[0]
    print(length)
    # np.arange 生成随机序列
    shuffle_indices = np.random.permutation(np.arange(length))
    #print(shuffle_indices)
    lines= lines[shuffle_indices]
    labels = labels[shuffle_indices]
    ii=0
    i=1
    print("write")
    while(ii<length):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'data': _bytes_feature(lines[ii]),
                    'label': _int64_feature(labels[ii])
        }
        )
        )
        if i%scale==0:
            writer2.write(example.SerializeToString())
        else:
            writer1.write(example.SerializeToString())

        i=i+1
        ii=ii+1
    writer1.close()
    writer2.close()
#随机读取tfrecord文件并生成数组区分sample和label
def ReadTFRecord(sqlpath,normalpath):
    # 读取文件生成队列
    filename_queue = tf.train.string_input_producer([sqlpath,normalpath], num_epochs=None)
    # 生成reader流
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'data':tf.FixedLenFeature([],tf.string),
                                           'label':tf.FixedLenFeature([],tf.int64)
                                       }
                                       )
    data_s=features['data']
    data = tf.decode_raw(features['data'], tf.int8)
    data_ar=data = tf.reshape(data, [80])
    data = tf.cast(data, tf.float32) * (1 / 255)
    label = tf.cast(features['label'], tf.int32)
    data_batch, label_batch = tf.train.shuffle_batch([data, label], batch_size=1,capacity=5000, min_after_dequeue=1000, num_threads=4)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)
    ii=0
    while(ii<2):
        data_val, label_val,data_ss,data_ar_s = sess.run([data_batch, label_batch,data_s,data_ar])
        ar = []
        str = data_ss.decode("utf8", "ignore")
        for ch in str:
            ar.append(int((ord(ch))))
        '''label_convert=[]
        i=0
        while i<len(label_val):
            if label_val[i] == 0:
                label_convert.append([0, 1])
            else:
                label_convert.append([1, 0])
            i=i+1
        label_val=label_convert'''
        print(data_ss)
        print(str)
        print(data_ar_s)
        print(ar)
        #print(label_val)
        ii=ii+1

if __name__ == '__main__':
    '''CreatTFRecordFile("E:\python-workspace\SQLExchange\length-normalize\\black-200-end.txt"\
                     ,"E:\python-workspace\SQLExchange\length-normalize\white-200.txt"\
                     ,"200.tfrecords")'''
    #CreatSingleTFRecordFile("use_data\\sql-80-test.txt","use_data\\normal-80-test.txt","tf_data\\80-test.tf")
    ConverTFRecordFile("length_data\\sql-ex-test-80.txt", "length_data\\sql-ex-test-80.tf", 0,80)
    ConverTFRecordFile("length_data\\sql-ex-test-120.txt", "length_data\\sql-ex-test-120.tf", 0,120)
    ConverTFRecordFile("length_data\\sql-ex-test-160.txt", "length_data\\sql-ex-test-160.tf", 0,160)
    ConverTFRecordFile("length_data\\sql-ex-test-200.txt", "length_data\\sql-ex-test-200.tf", 0,200)
    ConverTFRecordFile("length_data\\sql-ex-test-240.txt", "length_data\\sql-ex-test-240.tf", 0,240)
    ConverTFRecordFile("length_data\\sql-ex-test-280.txt", "length_data\\sql-ex-test-280.tf", 0,280)
    ConverTFRecordFile("length_data\\sql-ex-test-320.txt", "length_data\\sql-ex-test-320.tf", 0,320)
    ConverTFRecordFile("length_data\\sql-ex-test-360.txt", "length_data\\sql-ex-test-360.tf", 0,360)
    ConverTFRecordFile("length_data\\sql-ex-test-400.txt", "length_data\\sql-ex-test-400.tf", 0,400)

    ConverTFRecordFile("length_data\\sql-test-80.txt", "length_data\\sql-test-80.tf", 0,80)
    ConverTFRecordFile("length_data\\sql-test-120.txt", "length_data\\sql-test-120.tf", 0,120)
    ConverTFRecordFile("length_data\\sql-test-160.txt", "length_data\\sql-test-160.tf", 0,160)
    ConverTFRecordFile("length_data\\sql-test-200.txt", "length_data\\sql-test-200.tf", 0,200)
    ConverTFRecordFile("length_data\\sql-test-240.txt", "length_data\\sql-test-240.tf", 0,240)
    ConverTFRecordFile("length_data\\sql-test-280.txt", "length_data\\sql-test-280.tf", 0,280)
    ConverTFRecordFile("length_data\\sql-test-320.txt", "length_data\\sql-test-320.tf", 0,320)
    ConverTFRecordFile("length_data\\sql-test-360.txt", "length_data\\sql-test-360.tf", 0,360)
    ConverTFRecordFile("length_data\\sql-test-400.txt", "length_data\\sql-test-400.tf", 0,400)

    ConverTFRecordFile("length_data\\normal-ex-test-80.txt", "length_data\\normal-ex-test-80.tf", 1,80)
    ConverTFRecordFile("length_data\\normal-ex-test-120.txt", "length_data\\normal-ex-test-120.tf", 1,120)
    ConverTFRecordFile("length_data\\normal-ex-test-160.txt", "length_data\\normal-ex-test-160.tf", 1,160)
    ConverTFRecordFile("length_data\\normal-ex-test-200.txt", "length_data\\normal-ex-test-200.tf", 1,200)
    ConverTFRecordFile("length_data\\normal-ex-test-240.txt", "length_data\\normal-ex-test-240.tf", 1,240)
    ConverTFRecordFile("length_data\\normal-ex-test-280.txt", "length_data\\normal-ex-test-280.tf", 1,280)
    ConverTFRecordFile("length_data\\normal-ex-test-320.txt", "length_data\\normal-ex-test-320.tf", 1,320)
    ConverTFRecordFile("length_data\\normal-ex-test-360.txt", "length_data\\normal-ex-test-360.tf", 1,360)
    ConverTFRecordFile("length_data\\normal-ex-test-400.txt", "length_data\\normal-ex-test-400.tf", 1,400)

    ConverTFRecordFile("length_data\\normal-test-80.txt", "length_data\\normal-test-80.tf", 1,80)
    ConverTFRecordFile("length_data\\normal-test-120.txt", "length_data\\normal-test-120.tf", 1,120)
    ConverTFRecordFile("length_data\\normal-test-160.txt", "length_data\\normal-test-160.tf", 1,160)
    ConverTFRecordFile("length_data\\normal-test-200.txt", "length_data\\normal-test-200.tf", 1,200)
    ConverTFRecordFile("length_data\\normal-test-240.txt", "length_data\\normal-test-240.tf", 1,240)
    ConverTFRecordFile("length_data\\normal-test-280.txt", "length_data\\normal-test-280.tf", 1,280)
    ConverTFRecordFile("length_data\\normal-test-320.txt", "length_data\\normal-test-320.tf", 1,320)
    ConverTFRecordFile("length_data\\normal-test-360.txt", "length_data\\normal-test-360.tf", 1,360)
    ConverTFRecordFile("length_data\\normal-test-400.txt", "length_data\\normal-test-400.tf", 1,400)

    #ReadTFRecord("tf_data\\sql-80-train.tf","tf_data\\sql-80-train.tf")
    ConverTFRecordFile("length_data\\sql-ex-train-80.txt", "length_data\\sql-ex-train-80.tf", 0, 80)
    ConverTFRecordFile("length_data\\sql-ex-train-120.txt", "length_data\\sql-ex-train-120.tf", 0, 120)
    ConverTFRecordFile("length_data\\sql-ex-train-160.txt", "length_data\\sql-ex-train-160.tf", 0, 160)
    ConverTFRecordFile("length_data\\sql-ex-train-200.txt", "length_data\\sql-ex-train-200.tf", 0, 200)
    ConverTFRecordFile("length_data\\sql-ex-train-240.txt", "length_data\\sql-ex-train-240.tf", 0, 240)
    ConverTFRecordFile("length_data\\sql-ex-train-280.txt", "length_data\\sql-ex-train-280.tf", 0, 280)
    ConverTFRecordFile("length_data\\sql-ex-train-320.txt", "length_data\\sql-ex-train-320.tf", 0, 320)
    ConverTFRecordFile("length_data\\sql-ex-train-360.txt", "length_data\\sql-ex-train-360.tf", 0, 360)
    ConverTFRecordFile("length_data\\sql-ex-train-400.txt", "length_data\\sql-ex-train-400.tf", 0, 400)

    ConverTFRecordFile("length_data\\sql-train-80.txt", "length_data\\sql-train-80.tf", 0, 80)
    ConverTFRecordFile("length_data\\sql-train-120.txt", "length_data\\sql-train-120.tf", 0, 120)
    ConverTFRecordFile("length_data\\sql-train-160.txt", "length_data\\sql-train-160.tf", 0, 160)
    ConverTFRecordFile("length_data\\sql-train-200.txt", "length_data\\sql-train-200.tf", 0, 200)
    ConverTFRecordFile("length_data\\sql-train-240.txt", "length_data\\sql-train-240.tf", 0, 240)
    ConverTFRecordFile("length_data\\sql-train-280.txt", "length_data\\sql-train-280.tf", 0, 280)
    ConverTFRecordFile("length_data\\sql-train-320.txt", "length_data\\sql-train-320.tf", 0, 320)
    ConverTFRecordFile("length_data\\sql-train-360.txt", "length_data\\sql-train-360.tf", 0, 360)
    ConverTFRecordFile("length_data\\sql-train-400.txt", "length_data\\sql-train-400.tf", 0, 400)

    ConverTFRecordFile("length_data\\normal-ex-train-80.txt", "length_data\\normal-ex-train-80.tf", 1, 80)
    ConverTFRecordFile("length_data\\normal-ex-train-120.txt", "length_data\\normal-ex-train-120.tf", 1, 120)
    ConverTFRecordFile("length_data\\normal-ex-train-160.txt", "length_data\\normal-ex-train-160.tf", 1, 160)
    ConverTFRecordFile("length_data\\normal-ex-train-200.txt", "length_data\\normal-ex-train-200.tf", 1, 200)
    ConverTFRecordFile("length_data\\normal-ex-train-240.txt", "length_data\\normal-ex-train-240.tf", 1, 240)
    ConverTFRecordFile("length_data\\normal-ex-train-280.txt", "length_data\\normal-ex-train-280.tf", 1, 280)
    ConverTFRecordFile("length_data\\normal-ex-train-320.txt", "length_data\\normal-ex-train-320.tf", 1, 320)
    ConverTFRecordFile("length_data\\normal-ex-train-360.txt", "length_data\\normal-ex-train-360.tf", 1, 360)
    ConverTFRecordFile("length_data\\normal-ex-train-400.txt", "length_data\\normal-ex-train-400.tf", 1, 400)

    ConverTFRecordFile("length_data\\normal-train-80.txt", "length_data\\normal-train-80.tf", 1, 80)
    ConverTFRecordFile("length_data\\normal-train-120.txt", "length_data\\normal-train-120.tf", 1, 120)
    ConverTFRecordFile("length_data\\normal-train-160.txt", "length_data\\normal-train-160.tf", 1, 160)
    ConverTFRecordFile("length_data\\normal-train-200.txt", "length_data\\normal-train-200.tf", 1, 200)
    ConverTFRecordFile("length_data\\normal-train-240.txt", "length_data\\normal-train-240.tf", 1, 240)
    ConverTFRecordFile("length_data\\normal-train-280.txt", "length_data\\normal-train-280.tf", 1, 280)
    ConverTFRecordFile("length_data\\normal-train-320.txt", "length_data\\normal-train-320.tf", 1, 320)
    ConverTFRecordFile("length_data\\normal-train-360.txt", "length_data\\normal-train-360.tf", 1, 360)
    ConverTFRecordFile("length_data\\normal-train-400.txt", "length_data\\normal-train-400.tf", 1, 400)

    ''' CreatSingleTFRecordFile("E:\python-workspace\SQLExchange\length-normalize\\black-200-end.txt"\
                     ,"E:\python-workspace\SQLExchange\length-normalize\white-200.txt"\
                     ,"200.tfrecords")'''