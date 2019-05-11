#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import random

#sql训练集行数
sql_train_file_count=0
#对照训练集行数
normal_train_file_count=0
#sql测试集行数
sql_test_file_count=0
#对照测试集行数
normal_test_file_count=0
#训练sql取用行数
sql_batch_count=0
#训练对照取用行数
normal_batch_count=0
#测试sql取用行数
test_sql_batch_count=0
#测试对照取用行数
test_normal_batch_count=0

#训练集获取数据 txt版本 效率较慢
def get_train_batch_txt(sqlPath,normalPath,batch):
    global sql_train_file_count
    if sql_train_file_count<=0:
        for index, line in enumerate(open(sqlPath,'rb')):
            sql_train_file_count =sql_train_file_count+1
    global normal_train_file_count
    if normal_train_file_count<=0:
        for index, line in enumerate(open(normalPath,'rb')):
            normal_train_file_count =normal_train_file_count+1
    data=[]
    lable=[]
    global sql_batch_count
    global normal_batch_count
    if sql_batch_count+batch/2>sql_train_file_count:
        sql_batch_count=0
    if normal_batch_count+batch/2>normal_train_file_count:
        normal_batch_count=0

    for cur_line_number, line in enumerate(open(sqlPath,'rb'),start=int(sql_batch_count)):
        if cur_line_number >= sql_batch_count and cur_line_number < sql_batch_count+batch/2:
            ar=[]
            line=line.decode("utf8","ignore")
            for ch in line:
                ar.append(int((ord(ch))))
            data.append(ar)
            lable.append([0,1])
    for cur_line_number, line in enumerate(open(normalPath,'rb'),start=int(normal_batch_count)):
         if cur_line_number >= normal_batch_count and cur_line_number < normal_batch_count+batch/2:
            ar = []
            line=line.decode("utf8","ignore")
            for ch in line:
                ar.append(int((ord(ch))))
            data.append(ar)
            lable.append([1,0])
    data = np.asarray(data)
    lable = np.asarray(lable)
    # np.arange 生成随机序列
    shuffle_indices = np.random.permutation(np.arange(batch))
    data = data[shuffle_indices]
    lable = lable[shuffle_indices]
    sql_batch_count=sql_batch_count + batch/2
    normal_batch_count=normal_batch_count + batch / 2
    return data,lable
#测试集batch获取数据 txt版本 效率较慢
def get_test_batch_txt(sqlPath,normalPath,batch,length):
    global sql_test_file_count
    if sql_test_file_count<=0:
        for index, line in enumerate(open(sqlPath,'rb')):
            sql_test_file_count =sql_test_file_count+1
    global normal_test_file_count
    if normal_test_file_count<=0:
        for index, line in enumerate(open(normalPath,'rb')):
            normal_test_file_count =normal_test_file_count+1
    #i=0
    data=[]
    lable=[]
    global test_sql_batch_count
    global test_normal_batch_count
    if test_sql_batch_count+batch/2>sql_test_file_count:
        test_sql_batch_count=0
    if test_normal_batch_count+batch/2>normal_test_file_count:
        test_normal_batch_count=0
    i=0
    for cur_line_number, line in enumerate(open(sqlPath,'rb')):
        if i< batch/2:
            ar=[]
            if (cur_line_number < test_sql_batch_count):
                continue
            leng = len(line)
            if leng != length:
                continue
            # print(line)
            i = i + 1
            ii = 0
            while (ii < leng):
                ar.append(int(line[ii]))
                ii = ii + 1
            data.append(ar)
            lable.append([0,1])
    i=0
    for cur_line_number, line in enumerate(open(normalPath,'rb')):
         if i < batch/2:
            ar = []
            if (cur_line_number < test_normal_batch_count):
                continue
            leng = len(line)
            if leng != length:
                continue
            # print(line)
            i = i + 1
            ii = 0
            while (ii < leng):
                ar.append(int(line[ii]))
                ii = ii + 1
            data.append(ar)
            lable.append([1,0])
    #i=i+1
    data = np.asarray(data)
    lable = np.asarray(lable)
    # np.arange 生成随机序列
    shuffle_indices = np.random.permutation(np.arange(batch))
    data = data[shuffle_indices]
    lable = lable[shuffle_indices]
    test_sql_batch_count=test_sql_batch_count + batch/2
    test_normal_batch_count=test_normal_batch_count + batch / 2
    return data,lable

#测试集随机获取数据 txt版本 效率较慢
def get_test_random_txt(sqlPath,normalPath,batch,length):
    sql_count=0
    for index in enumerate(open(sqlPath,'rb')):
            sql_count =sql_count+1
    normal_count=0
    for index in enumerate(open(normalPath,'rb')):
            normal_count =normal_count+1
    #i=0
    data=[]
    lable=[]
    test_sql_batch=random.randint(0,sql_count)#在某个范围生成随机整数
    #print(test_sql_batch)
    test_normal_batch=random.randint(0,normal_count)
    #print(test_normal_batch)
    if test_sql_batch+batch/2>sql_count:
        test_sql_batch=0
    if test_normal_batch+batch/2>normal_count:
        test_normal_batch=0
    i=0
    for cur_line_number, line in enumerate(open(sqlPath,'rb')):
        if i< batch/2:
            ar=[]
            #line=line.decode("utf-8","ignore")
            if(cur_line_number<test_sql_batch):
                continue
            leng = len(line)
            if leng!=length:
                continue
            #print(line)
            i = i + 1
            ii=0
            while(ii<leng):
                ar.append(int(line[ii]))
                ii=ii+1
            data.append(ar)
            lable.append([0,1])
    i=0
    for cur_line_number, line in enumerate(open(normalPath,'rb')):
         if i < batch/2:
            ar = []
            #line=line.decode("utf-8","ignore")
            if (cur_line_number < test_normal_batch):
                continue
            leng = len(line)
            if leng != length:
                continue
            # print(line)
            i = i + 1
            ii = 0
            while (ii < leng):
                ar.append(int(line[ii]))
                ii = ii + 1
            data.append(ar)
            lable.append([1,0])
    #i=i+1
    data = np.asarray(data)
    lable = np.asarray(lable)
    # np.arange 生成随机序列
    shuffle_indices = np.random.permutation(np.arange(batch))
    data = data[shuffle_indices]
    lable = lable[shuffle_indices]
    return data,lable

#从tf格式文件中获取batch数据 分别从sql normal文件中获取 ,效率低
def get_batch_tf(sqlPath,normalPath,length,batch):
    sessr = tf.Session()
    # 读取文件生成队列
    filename_queue1 = tf.train.string_input_producer([sqlPath], num_epochs=None)
    filename_queue2 = tf.train.string_input_producer([normalPath], num_epochs=None)
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
    data1 = tf.reshape(data1, [length])
    data1 = tf.cast(data1, tf.float32) * (1 / 255)
    label1 = tf.cast(features1['label'], tf.int32)
    data_batch1, label_batch1 = tf.train.shuffle_batch([data1, label1], batch_size=int(batch/2), capacity=500,
                                                     min_after_dequeue=200, num_threads=4)
    # 转换数据 normal
    data2 = tf.decode_raw(features2['data'], tf.int8)
    data2 = tf.reshape(data2, [length])
    data2 = tf.cast(data2, tf.float32) * (1 / 255)
    label2 = tf.cast(features2['label'], tf.int32)
    data_batch2, label_batch2 = tf.train.shuffle_batch([data2, label2], batch_size=int(batch / 2), capacity=500,
                                                       min_after_dequeue=200, num_threads=4)
    sessr.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sessr)
    data_val1, label_val1 = sessr.run([data_batch1, label_batch1])
    data_val2, label_val2 = sessr.run([data_batch2, label_batch2])
    data_convert = []
    label_convert = []
    i = 0
    while i < int(batch / 2):
        if label_val1[i] == 0:
            label_convert.append([0, 1])
        else:
            label_convert.append([1, 0])
        data_convert.append(data_val1[i])
        if label_val2[i] == 0:
            label_convert.append([0, 1])
        else:
            label_convert.append([1, 0])
        data_convert.append(data_val2[i])
        i = i + 1

    data_convert = np.asarray(data_convert)
    label_convert = np.asarray(label_convert)
    # np.arange 生成随机序列
    shuffle_indices = np.random.permutation(np.arange(batch))
    data_convert = data_convert[shuffle_indices]
    label_convert = label_convert[shuffle_indices]
    sessr.close()
    return data_convert,label_convert


#测试集随机获取数据 txt版本 效率较慢
def get_test_random_txt(sqlPath,normalPath,batch,length):
    sql_count=0
    for index in enumerate(open(sqlPath,'rb')):
            sql_count =sql_count+1
    normal_count=0
    for index in enumerate(open(normalPath,'rb')):
            normal_count =normal_count+1
    #i=0
    data=[]
    lable=[]
    test_sql_batch=random.randint(0,sql_count)#在某个范围生成随机整数
    #print(test_sql_batch)
    test_normal_batch=random.randint(0,normal_count)
    #print(test_normal_batch)
    if test_sql_batch+batch/2>sql_count:
        test_sql_batch=0
    if test_normal_batch+batch/2>normal_count:
        test_normal_batch=0
    i=0
    for cur_line_number, line in enumerate(open(sqlPath,'rb')):
        if i< batch/2:
            ar=[]
            #line=line.decode("utf-8","ignore")
            if(cur_line_number<test_sql_batch):
                continue
            leng = len(line)
            if leng!=length:
                continue
            #print(line)
            i = i + 1
            ii=0
            while(ii<leng):
                ar.append(int(line[ii]))
                ii=ii+1
            data.append(ar)
            lable.append([0,1])
    i=0
    for cur_line_number, line in enumerate(open(normalPath,'rb')):
         if i < batch/2:
            ar = []
            #line=line.decode("utf-8","ignore")
            if (cur_line_number < test_normal_batch):
                continue
            leng = len(line)
            if leng != length:
                continue
            # print(line)
            i = i + 1
            ii = 0
            while (ii < leng):
                ar.append(int(line[ii]))
                ii = ii + 1
            data.append(ar)
            lable.append([1,0])
    #i=i+1
    data = np.asarray(data)
    lable = np.asarray(lable)
    # np.arange 生成随机序列
    shuffle_indices = np.random.permutation(np.arange(batch))
    data = data[shuffle_indices]
    lable = lable[shuffle_indices]
    return data,lable
#测试集获取，根据输入的标签返回样本
def get_test_random_txt_single(Path,la,batch,length):
    count=0
    for index in enumerate(open(Path,'rb')):
            count =count+1
    #i=0
    data=[]
    lable=[]
    test_batch=random.randint(0,count)#在某个范围生成随机整数
    #print(test_normal_batch)
    if test_batch+batch>count:
        test_batch=0
    print(test_batch)
    i=0
    for cur_line_number, line in enumerate(open(Path,'rb')):
        if i< batch:
            ar=[]
            #line=line.decode("utf-8","ignore")
            if(cur_line_number<test_batch):
                continue
            leng = len(line)
            if leng!=length:
                continue
            #print(line)
            i = i + 1
            ii=0
            while(ii<leng):
                ar.append(int(line[ii]))
                ii=ii+1
            data.append(ar)
            if la==0:
                lable.append([0,1])
            else:
                lable.append([1,0])
    data = np.asarray(data)
    lable = np.asarray(lable)
    # np.arange 生成随机序列
    shuffle_indices = np.random.permutation(np.arange(batch))
    data = data[shuffle_indices]
    lable = lable[shuffle_indices]
    return data,lable,count
#测试集获取，根据输入的标签返回样本
def get_test_all_txt_single(Path,la,length):
    count=0
    data=[]
    strline=[]
    lable=[]
    for cur_line_number, line in enumerate(open(Path,'rb')):
        ar=[]
        leng = len(line)
        if leng!=length:
            continue
        count = count + 1
        strline.append(line)
        ii=0
        while(ii<leng):
            ar.append(int(line[ii]))
            ii=ii+1
        data.append(ar)
        if la==0:
            lable.append([0,1])
        else:
            lable.append([1,0])
    return data,lable,count,strline