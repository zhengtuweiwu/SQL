#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import re
from urllib.parse import unquote
import random
import numpy as np
#循环读取目录下所有文件名并返回文件名队列
def ReadDir(dirPath):
    if dirPath != "":
        files = os.listdir(dirPath)
        path=[]
        for file in files:
            file=dirPath+"\\" +file
            path.append(file)
            #print(dirPath + "/" + file)
        return path
    else:
        return 0
#将逐行文件拷贝至单一文件中
def CopyToFile(sPath,dPath):
    read_f = open(sPath,'rb')  # 打开原始文件
    write_f = open(dPath, 'ab') # 打开目标文件
    for line in read_f.readlines():
        line=line.strip()
        write_f.write(line+b'\n')  #确保读文件的正确性，带换行符，否则datan打不开
        #print(line)  # 打印一行
    read_f.close()
    write_f.close()
    return 0
#清除文件中包含的空格，逐行清除
def SubSpace(sPath,dPath):
    read_f = open(sPath,'rb')  # 打开原始文件
    write_f = open(dPath, 'ab') # 打开目标文件
    for line in read_f.readlines():
        #re.sub(r"\s+", "", line)
        line=line.decode("utf8","ignore").replace(" ","")
        write_f.write(line.encode("utf8","ignore"))  #确保读文件的正确性，带换行符，否则datan打不开
        #print(line)  # 打印一行
    read_f.close()
    write_f.close()
    return 0
#逐行清除url类转义字符并保存文件
def ExchangeURLCode(sPath,dPath):
    read_f = open(sPath,'rb')  # 打开原始文件
    write_f = open(dPath,'ab') # 打开目标文件
    for line in read_f.readlines():
        '''line=line.decode("utf8","ignore").replace("%20"," ").replace("%22","\"").replace("%23","#").replace("%25","%")\
            .replace("%26","&").replace("%28","(").replace("%29",")").replace("%2B","+") \
            .replace("%2C", ",").replace("%2F","/").replace("%3A",":").replace("%3B",";") \
            .replace("%3C", "<").replace("%3D","=").replace("%3E",">").replace("%3F","?") \
            .replace("%40", "@").replace("%5C","\\").replace("%7C","|").replace("%27","'") \
            .replace("%24", "$").replace("%2A","*").replace("%21","!").replace("%2D","-") \
            .replace("%2E", ".").replace("%30","`")#替换转义字符'''
        line = line.decode("utf8", "ignore")
        line=unquote(line, 'utf-8','replace')
        write_f.write(str.encode(line))  #此时不需要多加换行符
        #print(line)  # 打印一行
    read_f.close()
    write_f.close()
    return 0

# 逐行转换0x16进制字符为FF表示 转换十进制数字为DD表示
def ExchangeFFDDCode(sPath,dPath):
    read_f = open(sPath, 'rb')  # 打开原始文件
    write_f = open(dPath, 'ab')  # 打开目标文件
    for line in read_f.readlines():
        line = re.sub(r"0x[a-fA-F0-9]+","0xFF",line.decode("utf8","ignore")) #替换十六进制
        line = re.sub(r"[0-9]+", "DD", line) #替换十进制数字
        write_f.write(line.encode("utf8","ignore"))  ##此时不需要多加换行符
        # print(line)  # 打印一行
    read_f.close()
    write_f.close()
    return 0
#文件中文本行去重
def SubRepeatLine(sPath,dPath):
    read_f = open(sPath,"rb")
    original_list = read_f.readlines() #读取全部内容 ，并以列表方式返回
    print(len(original_list))
    read_f.close()
    newlist=[]
    for iline in original_list:	#遍历去重
        if not iline.decode("utf8","ignore") in newlist:
            newlist.append(iline.decode("utf8","ignore"))
    print(len(newlist))
    newtxt="".join(newlist)
    write_f = open(dPath,"wb")
    write_f.write(newtxt.encode("utf8","ignore"))
    write_f.close()
#统计每行的长度并按照数量进行排序打印
def LengthStatic(path):
    read_f = open(path, "rb")
    original_list = read_f.readlines()  # 读取全部内容 ，并以列表方式返回
    read_f.close()
    len_list = []
    for iline in original_list:
        len_list.append(len(iline)) #生成长度list
    dict = {}
    dict_len_quare = {}
    for key in len_list:
        dict[key] = dict.get(key, 0) + 1
        if key<=40:
            dict_len_quare["小于40"]=dict_len_quare.get("小于40", 0) + 1
        if key>40 and key<=80:
            dict_len_quare["大于40小于80"]=dict_len_quare.get("大于40小于80", 0) + 1
        if key>80 and key<=160:
            dict_len_quare["大于80小于160"]=dict_len_quare.get("大于80小于160", 0) + 1
        if key>160 and key<=240:
            dict_len_quare["大于160小于240"]=dict_len_quare.get("大于160小于240", 0) + 1
        if key>240 and key<=320:
            dict_len_quare["大于240小于320"]=dict_len_quare.get("大于240小于320", 0) + 1
        if key>320 and key<=400:
            dict_len_quare["大于320小于400"]=dict_len_quare.get("大于320小于400", 0) + 1
        if key>400:
            dict_len_quare["大于400"]=dict_len_quare.get("大于400", 0) + 1
    sorted(dict.keys())  #字典按照key值重新排列顺序
    #sorted(dict_len_quare.keys())
    print(len(len_list))
    print(dict)
    print("小于40: "+str(dict_len_quare.get("小于40",0)))
    print("大于40小于80: " + str(dict_len_quare.get("大于40小于80", 0)))
    print("大于80小于160: " + str(dict_len_quare.get("大于80小于160", 0)))
    print("大于160小于240: " + str(dict_len_quare.get("大于160小于240", 0)))
    print("大于240小于320: " + str(dict_len_quare.get("大于240小于320", 0)))
    print("大于320小于400: " + str(dict_len_quare.get("大于320小于400", 0)))
    print("大于400: " + str(dict_len_quare.get("大于400", 0)))
    return 0
#统计SQL语句特征字段占比
def StaticsSQLFilter(path):
    read_f = open(path, "rb")
    original_list = read_f.readlines()  # 读取全部内容 ，并以列表方式返回
    read_f.close()
    dict={}
    for iline in original_list:
        iline=iline.decode("utf8","ignore")
        if re.search(r"[sS][eE][lL][eE][cC][tT]", iline):
            dict["select"] = dict.get("select", 0) + 1
        if re.search(r"[oO][rR]", iline):
            dict["or"] = dict.get("or", 0) + 1
        if re.search(r"[aA][nN][dD]", iline):
            dict["and"] = dict.get("and", 0) + 1
        if re.search(r"[aA][lL]{2}", iline):
            dict["all"] = dict.get("all", 0) + 1
        if re.search(r"[sS][eE][tT]", iline):
            dict["set"] = dict.get("set", 0) + 1
        if re.search(r"[aA][lL][tT][eE][rR]", iline):
            dict["alter"] = dict.get("alter", 0) + 1
        if re.search(r"[dD][eE][cC][lL][aA][rR][eE]", iline):
            dict["declare"] = dict.get("declare", 0) + 1
        if re.search(r"[cC][aA][sS][tT]", iline):
            dict["cast"] = dict.get("cast", 0) + 1
        if re.search(r"[fF][rR][oO][mM]", iline):
            dict["from"] = dict.get("from", 0) + 1
        if re.search(r"[lL][iI][kK][eE]", iline):
            dict["like"] = dict.get("like", 0) + 1
        if re.search(r"[vV][eE][rR][sS][iI][oO][nN]", iline):
            dict["version"] = dict.get("version", 0) + 1
        if re.search(r"[lL][eE][nN][gG][tT][hH]", iline):
            dict["length"] = dict.get("length", 0) + 1
        if re.search(r"[wW][hH][eE][rR][eE]", iline):
            dict["where"] = dict.get("where", 0) + 1
        if re.search(r"[uU][nN][iI][oO][nN]", iline):
            dict["union"] = dict.get("union", 0) + 1
        if re.search(r"[cC][hH][aA][rR]", iline):
            dict["char"] = dict.get("char", 0) + 1
        if re.search(r"[pP][aA][sS]{2}[wW][oO][rR][dD]", iline):
            dict["password"] = dict.get("password", 0) + 1
        if re.search(r"[dD][rR][oO][pP]", iline):
            dict["drop"] = dict.get("drop", 0) + 1
        if re.search(r"[tT][aA][bB][lL][eE]", iline):
            dict["table"] = dict.get("table", 0) + 1
        if re.search(r"[iI][nN][tT][oO]", iline):
            dict["into"] = dict.get("into", 0) + 1
        if re.search(r"[pP][aA][sS]{2}[wW][dD]", iline):
            dict["passwd"] = dict.get("passwd", 0) + 1
        if re.search(r"[sS][lL][eE]{2}[pP]", iline):
            dict["sleep"] = dict.get("sleep", 0) + 1
        if re.search(r"[bB][iI][nN][aA][rR][Yy]", iline):
            dict["binary"] = dict.get("binary", 0) + 1
        if re.search(r"[cC][rR][eE][aA][tT]", iline):
            dict["creat"] = dict.get("creat", 0) + 1
        if re.search(r"[wW][hH][eE][nN]", iline):
            dict["when"] = dict.get("when", 0) + 1
        if re.search(r"[gG][rR][oO][uU][pP]", iline):
            dict["group"] = dict.get("group", 0) + 1
        if re.search(r"[eE][xX][eE][cC]", iline):
            dict["exec"] = dict.get("exec", 0) + 1
    print(len(original_list))
    dict=sorted(dict.items(),key = lambda x:x[1],reverse = True) #按数值重新排序后生成字典
    print(dict)
#清除超短行 length长度
def SubShortLine(sPath,dPath,length):
    read_f = open(sPath, "rb")
    write_f = open(dPath, "wb")
    for line in read_f.readlines():  # 遍历
        if len(line)>length:
            write_f.write(line)  ##此时不需要多加换行符
    write_f.close()
    read_f.close()
    return 0
#分配比例抽取文件 scale为随机抽取数
def CutoffFile(sPath,dPath,scale):
    read_f = open(sPath, "rb")
    write_f = open(dPath, "wb")
    i=0;
    for line in read_f.readlines():  # 遍历
        i=i+1
        if (i%scale==0):
            write_f.write(line)  ##此时不需要多加换行符
    write_f.close()
    read_f.close()
    return 0
#按照输入长度对文本行进行长度归一化操作 补齐000000或者裁剪  此处转换有缺陷 utf-8编码在汉字处理上解码编码时会产生数据流长度变化问题
def LengthNormalize(sPath,dPath,length):
    read_f = open(sPath, "rb")
    write_f = open(dPath, "wb")
    for line in read_f.readlines():  # 遍历
        lenline=len(line)
        line=line.decode("utf8","ignore")
        if lenline > length:
            i=0
            ss=lenline-length
            line.replace('\n','')
            while(i<=ss):
                write_f.write(line[i:(i+length-1)].encode("utf8","ignore")+b'\n')
                i=i+1
        elif lenline == length:
            write_f.write(line.encode("utf8","ignore"))  #此时不需要多加换行符
        else:
            ll=length-lenline
            i=0
            line=line.replace('\n', '')
            while(i<=ll):
                ii=0
                newlint=""
                while(ii<i):
                    newlint=newlint+"0"
                    ii=ii+1
                newlint=newlint+line
                iii=0
                while(iii<ll-i):
                    newlint=newlint+"0"
                    iii=iii+1
                write_f.write(newlint.encode("utf8", "ignore")+b'\n')
                i=i+1

    write_f.close()
    read_f.close()
    return 0
#按照输入byte长度对文本行进行长度归一化操作 补齐000000或者裁剪 无编码问题
def LengthNormalize_byte(sPath,dPath,length):
    read_f = open(sPath, "rb")
    write_f = open(dPath, "wb")
    for line in read_f.readlines():  # 遍历
        lenline=len(line)
        if lenline > length:
            i=0
            ss=lenline-length
            while(i<=ss):
                #ii=0
                convertdata=b""
                '''while(ii<length-1):
                    convertdata=convertdata+chr(line[i+ii]).encode("utf8","ignore")
                    ii=ii+1'''
                convertdata=convertdata+line[i:i+length-1]
                #convertdata = convertdata + (b'\n').decode("utf8","ignore")
                convertdata = convertdata + b'\n'
                if len(convertdata)==length:
                    write_f.write(convertdata)
                i=i+1
        elif lenline == length:
            write_f.write(line)  #此时不需要多加换行符
        else:
            ll=length-lenline
            i=0
            while(i<=ll):
                ii=0
                newlint=b""
                while(ii<i):
                    newlint=newlint+b'0'
                    ii=ii+1
                '''iiii=0
                while(iiii<lenline-1):
                    newlint=newlint+chr(line[iiii]).encode("utf8","ignore")
                    iiii=iiii+1'''
                newlint = newlint + line[0:lenline - 1]
                iii=0
                while(iii<ll-i):
                    newlint=newlint+b'0'
                    iii=iii+1
                newlint=newlint+ b'\n'
                if len(newlint)==length:
                    write_f.write(newlint)
                i=i+1

    write_f.close()
    read_f.close()
    return 0
def LengthFilter(sPath,dPath,length):
    read_f = open(sPath, "rb")
    write_f = open(dPath, "wb")
    i=0
    for line in read_f.readlines():  # 遍历
        lenline=len(line)
        #line=line.decode("utf8","ignore")
        if lenline == length:
            write_f.write(line)  #此时不需要多加换行符
        else:
            i=i+1
    print(i)
    write_f.close()
    read_f.close()
    return 0
#清除无关键过滤词的归一化后的文件中的行，该操作只对长度归一化后的SQL注入样本进行
def SubNoSQLFilterLine(sPath,dPath):
    read_f = open(sPath, "rb")
    write_f = open(dPath, "wb")
    for iline in read_f.readlines():  # 遍历
        iline=iline.decode("utf8","ignore")
        if re.search(r"[sS][eE][lL][eE][cC][tT]", iline) or re.search(r"[oO][rR]", iline) \
                or re.search(r"[aA][nN][dD]", iline) or re.search(r"[aA][lL]{2}", iline) \
                or re.search(r"[sS][eE][tT]", iline) or re.search(r"[aA][lL][tT][eE][rR]", iline) \
                or re.search(r"[dD][eE][cC][lL][aA][rR][eE]", iline) or re.search(r"[cC][aA][sS][tT]", iline) \
                or re.search(r"[fF][rR][oO][mM]", iline) or re.search(r"[lL][iI][kK][eE]", iline) \
                or re.search(r"[vV][eE][rR][sS][iI][oO][nN]", iline) or re.search(r"[lL][eE][nN][gG][tT][hH]", iline) \
                or re.search(r"[wW][hH][eE][rR][eE]", iline) or re.search(r"[uU][nN][iI][oO][nN]", iline) \
                or re.search(r"[cC][hH][aA][rR]", iline) or re.search(r"[pP][aA][sS]{2}[wW][oO][rR][dD]", iline) \
                or re.search(r"[dD][rR][oO][pP]", iline) or re.search(r"[tT][aA][bB][lL][eE]", iline) \
                or re.search(r"[iI][nN][tT][oO]", iline) or re.search(r"[pP][aA][sS]{2}[wW][dD]", iline) \
                or re.search(r"[sS][lL][eE]{2}[pP]", iline) or re.search(r"[bB][iI][nN][aA][rR][Yy]", iline) \
                or re.search(r"[cC][rR][eE][aA][tT]", iline) or re.search(r"[wW][hH][eE][nN]", iline) \
                or re.search(r"[gG][rR][oO][uU][pP]", iline) or re.search(r"[eE][xX][eE][cC]", iline):
            write_f.write(iline.encode("utf8", "ignore"))  ##此时不需要多加换行符
    write_f.close()
    read_f.close()
    return 0
#输出sql非过滤词样本
def SubSQLFilterLine(sPath,dPath):
    read_f = open(sPath, "rb")
    write_f = open(dPath, "wb")
    for iline in read_f.readlines():  # 遍历
        iline=iline.decode("utf8","ignore")
        if re.search(r"[sS][eE][lL][eE][cC][tT]", iline) or re.search(r"[oO][rR]", iline) \
                or re.search(r"[aA][nN][dD]", iline) or re.search(r"[aA][lL]{2}", iline) \
                or re.search(r"[sS][eE][tT]", iline) or re.search(r"[aA][lL][tT][eE][rR]", iline) \
                or re.search(r"[dD][eE][cC][lL][aA][rR][eE]", iline) or re.search(r"[cC][aA][sS][tT]", iline) \
                or re.search(r"[fF][rR][oO][mM]", iline) or re.search(r"[lL][iI][kK][eE]", iline) \
                or re.search(r"[vV][eE][rR][sS][iI][oO][nN]", iline) or re.search(r"[lL][eE][nN][gG][tT][hH]", iline) \
                or re.search(r"[wW][hH][eE][rR][eE]", iline) or re.search(r"[uU][nN][iI][oO][nN]", iline) \
                or re.search(r"[cC][hH][aA][rR]", iline) or re.search(r"[pP][aA][sS]{2}[wW][oO][rR][dD]", iline) \
                or re.search(r"[dD][rR][oO][pP]", iline) or re.search(r"[tT][aA][bB][lL][eE]", iline) \
                or re.search(r"[iI][nN][tT][oO]", iline) or re.search(r"[pP][aA][sS]{2}[wW][dD]", iline) \
                or re.search(r"[sS][lL][eE]{2}[pP]", iline) or re.search(r"[bB][iI][nN][aA][rR][Yy]", iline) \
                or re.search(r"[cC][rR][eE][aA][tT]", iline) or re.search(r"[wW][hH][eE][nN]", iline) \
                or re.search(r"[gG][rR][oO][uU][pP]", iline) or re.search(r"[eE][xX][eE][cC]", iline):
            pp=0
        else:
            write_f.write(iline.encode("utf8", "ignore"))  ##此时不需要多加换行符
            #write_f.write(iline.encode("utf8", "ignore"))  ##此时不需要多加换行符
    write_f.close()
    read_f.close()
    return 0
#清除无关键过滤词的归一化后的文件中的行，该操作只对长度归一化后的XSS注入样本进行
def SubNoXSSFilterLine(sPath,dPath):
    read_f = open(sPath, "rb")
    write_f = open(dPath, "wb")
    for iline in read_f.readlines():  # 遍历
        iline=iline.decode("utf8","ignore")
        if re.search(r"[aA][lL][eE][rR][tT]", iline):
            write_f.write(iline.encode("utf8", "ignore"))  ##此时不需要多加换行符
    write_f.close()
    read_f.close()
    return 0

#分配比例生成训练集样本集 scale为随机抽取数
def SampleTrainFile(sPath,dPath,scale):
    read_f = open(sPath, "rb")
    write_test = open(dPath+"-test.txt", "wb")
    write_train = open(dPath+"-train.txt", "wb")
    i=0;
    for line in read_f.readlines():  # 遍历
        i=i+1
        if (i%scale==0):
            write_test.write(line)  ##此时不需要多加换行符
        else:
            write_train.write(line)  ##此时不需要多加换行符
    write_test.close()
    write_train.close()
    read_f.close()
    return 0
###测试读取文件行byte数组转换问题
def readByte(path,strlen):
    ar=[]
    read_f = open(path, "rb")
    #i=0
    count=0
    icount=0
    for line in read_f.readlines():  # 遍历
        count=count+1
        #print(line)
        byte=[]
        ii=0
        while(ii < len(line)):
            #print(line[i])
            #str= unpack("idh", line.read(1))
            byte.append(line[ii])
            ii=ii+1
        if(len(byte)!=strlen):
            icount=icount+1
        #print(byte)
        #i=i+1
        #if i>100:
            #break
    print(count)
    print(icount)
    return 0

#sql测试集行数
sql_test_file_count=0
#对照测试集行数
normal_test_file_count=0

#测试sql取用行数
test_sql_batch_count=0
#测试对照取用行数
test_normal_batch_count=0
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

if __name__ == '__main__':
    '''filePaths=ReadDir("E:\\shell dataset\\normal-collect")
    if filePaths!=0:
        for filePath in filePaths:
            #print(filePath)
            CopyToFile(filePath,"nolength_data\\normal-original.txt")
            #break
    else:
        print("Directory Empty!")'''#合并成大文件并去除首尾多余空格

    #ExchangeURLCode("nolength_data\\normal-original-noshort-norepeat.txt","nolength_data\\normal-original-noshort-norepeat-urlexchange.txt") #url转义字符处理后的文件
    #ExchangeFFDDCode("nolength_data\\normal-original-noshort-norepeat.txt","nolength_data\\normal-original-noshort-norepeat-FFDD.txt") #十六进制转换成0xFF 十进制转换成DD
    #SubSpace("Third-Step.txt","Fourth-Step.txt") #去除全部的空格位
    #SubRepeatLine("nolength_data\\normal-original-noshort-norepeat-FFDD.txt","nolength_data\\normal-original-noshort-norepeat-FFDD-norepeat.txt") #去除重复的文本行
    #CutoffFile("nolength_data\\normal-original-noshort-norepeat-FFDD-norepeat.txt","nolength_data\\normal-original-noshort-norepeat-FFDD-norepeat-cutxss10.txt",10)#按照比例提取文件
    LengthStatic(r"nolength_data\\sql-original-noshort-norepeat-urlexchange-FFDD-norepeat-sqlfilter.txt") #统计每行的长度并按照数量进行排序打印
    #StaticsSQLFilter(r"nolength_data\\normal-original-noshort-norepeat-urlexchange-FFDD-norepeat-sqlfilter.txt")#统计SQL语句特征字段占比
    #SubShortLine("nolength_data\\normal-original.txt","nolength_data\\normal-original-noshort.txt",5)

    ####根据SQL-CNN网络实际构造，样本归一化的长度必须能被4整除--------80,120,160,200,240,280,320,360,400
    #SubNoSQLFilterLine("nolength_data\\normal-original-noshort-norepeat-urlexchange-FFDD-norepeat.txt","nolength_data\\normal-original-noshort-norepeat-urlexchange-FFDD-norepeat-filter.txt")#清除无关键过滤词的归一化后的文件中的行，该操作只对长度归一化后的SQL注入样本进行
    #SubSQLFilterLine("nolength_data\\sql-original-noshort-norepeat-urlexchange-FFDD-norepeat.txt","nolength_data\\sql-original-noshort-norepeat-urlexchange-FFDD-norepeat-nofilter.txt")#清除含有关键过滤词归一化后的文件
    ######xss过滤词还不够完善
    #SubNoXSSFilterLine("nolength_data\\xss-original-noshort-norepeat-urlexchange-FFDD-norepeat.txt","nolength_data\\xss-original-noshort-norepeat-urlexchange-FFDD-norepeat-filter.txt")  # 清除无关键过滤词的归一化后的文件中的行，该操作只对长度归一化后的SQL注入样本进行
    #LengthNormalize("nolength_data\\normal-original-noshort-norepeat-urlexchange-FFDD-norepeat.txt", "test_data\\normal-ex-80.txt",80)  # 按照输入长度对文本行进行长度归一化操作 补齐000000或者裁剪
    #SampleTrainFile("test_data\\normal-ex-80.txt", "test_data\\normal-ex-80", 10)
    #LengthFilter("test_data\\normal-ex-80-train.txt","test_data\\normal-ex-80-train-lenfilter.txt",80) #删除不符合长度的行

    #LengthFilter("test_data\\normal-ex-240-test-lenfilter.txt", "test_data\\normal-ex-240-test.txt", 240)  # 删除不符合长度的行
    #data,label=get_test_batch_txt("test_data\\sql-ex-240-test-lenfilter.txt","test_data\\normal-ex-240-test-lenfilter.txt",2,240)
    #print(data)
    #ExchangeURLCode("test-sql-like.txt","test-sql-like-url.txt") #url转义字符处理后的文件
    #ExchangeFFDDCode("test-sql-like-url.txt","test-sql-like-url-FFDD.txt")
    '''SampleTrainFile("sql-original-noshort-norepeat-urlexchange-FFDD-norepeat-sqlfilter.txt", "sql-ex", 10)
    LengthNormalize_byte("sql-ex-train.txt","test-80.txt",80)
    SubNoSQLFilterLine("test-80.txt", "sql-ex-train-80.txt")
    LengthNormalize_byte("sql-ex-test.txt", "test-80.txt", 80)
    SubNoSQLFilterLine("test-80.txt","sql-ex-test-80.txt")
    LengthNormalize_byte("sql-ex-train.txt", "test-120.txt", 120)
    SubNoSQLFilterLine("test-120.txt", "sql-ex-train-120.txt")
    LengthNormalize_byte("sql-ex-test.txt", "test-120.txt", 120)
    SubNoSQLFilterLine("test-120.txt", "sql-ex-test-120.txt")
    LengthNormalize_byte("sql-ex-train.txt", "test-160.txt", 160)
    SubNoSQLFilterLine("test-160.txt", "sql-ex-train-160.txt")
    LengthNormalize_byte("sql-ex-test.txt", "test-160.txt", 160)
    SubNoSQLFilterLine("test-160.txt", "sql-ex-test-160.txt")
    LengthNormalize_byte("sql-ex-train.txt", "test-200.txt", 200)
    SubNoSQLFilterLine("test-200.txt", "sql-ex-train-200.txt")
    LengthNormalize_byte("sql-ex-test.txt", "test-200.txt", 200)
    SubNoSQLFilterLine("test-200.txt", "sql-ex-test-200.txt")
    LengthNormalize_byte("sql-ex-train.txt", "test-240.txt", 240)
    SubNoSQLFilterLine("test-240.txt", "sql-ex-train-240.txt")
    LengthNormalize_byte("sql-ex-test.txt", "test-240.txt", 240)
    SubNoSQLFilterLine("test-240.txt", "sql-ex-test-240.txt")
    LengthNormalize_byte("sql-ex-train.txt", "test-280.txt", 280)
    SubNoSQLFilterLine("test-280.txt", "sql-ex-train-280.txt")
    LengthNormalize_byte("sql-ex-test.txt", "test-280.txt", 280)
    SubNoSQLFilterLine("test-280.txt", "sql-ex-test-280.txt")
    LengthNormalize_byte("sql-ex-train.txt", "test-320.txt", 320)
    SubNoSQLFilterLine("test-320.txt", "sql-ex-train-320.txt")
    LengthNormalize_byte("sql-ex-test.txt", "test-320.txt", 320)
    SubNoSQLFilterLine("test-320.txt", "sql-ex-test-320.txt")
    LengthNormalize_byte("sql-ex-train.txt", "test-360.txt", 360)
    SubNoSQLFilterLine("test-360.txt", "sql-ex-train-360.txt")
    LengthNormalize_byte("sql-ex-test.txt", "test-360.txt", 360)
    SubNoSQLFilterLine("test-360.txt", "sql-ex-test-360.txt")
    LengthNormalize_byte("sql-ex-train.txt", "test-400.txt", 400)
    SubNoSQLFilterLine("test-400.txt", "sql-ex-train-400.txt")
    LengthNormalize_byte("sql-ex-test.txt", "test-400.txt", 400)
    SubNoSQLFilterLine("test-400.txt", "sql-ex-test-400.txt")


    SampleTrainFile("sql-original-noshort-norepeat-sqlfilter.txt", "sql", 10)
    LengthNormalize_byte("sql-train.txt", "test-80.txt", 80)
    SubNoSQLFilterLine("test-80.txt", "sql-train-80.txt")
    LengthNormalize_byte("sql-test.txt", "test-80.txt", 80)
    SubNoSQLFilterLine("test-80.txt", "sql-test-80.txt")
    LengthNormalize_byte("sql-train.txt", "test-120.txt", 120)
    SubNoSQLFilterLine("test-120.txt", "sql-train-120.txt")
    LengthNormalize_byte("sql-test.txt", "test-120.txt", 120)
    SubNoSQLFilterLine("test-120.txt", "sql-test-120.txt")
    LengthNormalize_byte("sql-train.txt", "test-160.txt", 160)
    SubNoSQLFilterLine("test-160.txt", "sql-train-160.txt")
    LengthNormalize_byte("sql-test.txt", "test-160.txt", 160)
    SubNoSQLFilterLine("test-160.txt", "sql-test-160.txt")
    LengthNormalize_byte("sql-train.txt", "test-200.txt", 200)
    SubNoSQLFilterLine("test-200.txt", "sql-train-200.txt")
    LengthNormalize_byte("sql-test.txt", "test-200.txt", 200)
    SubNoSQLFilterLine("test-200.txt", "sql-test-200.txt")
    LengthNormalize_byte("sql-train.txt", "test-240.txt", 240)
    SubNoSQLFilterLine("test-240.txt", "sql-train-240.txt")
    LengthNormalize_byte("sql-test.txt", "test-240.txt", 240)
    SubNoSQLFilterLine("test-240.txt", "sql-test-240.txt")
    LengthNormalize_byte("sql-train.txt", "test-280.txt", 280)
    SubNoSQLFilterLine("test-280.txt", "sql-train-280.txt")
    LengthNormalize_byte("sql-test.txt", "test-280.txt", 280)
    SubNoSQLFilterLine("test-280.txt", "sql-test-280.txt")
    LengthNormalize_byte("sql-train.txt", "test-320.txt", 320)
    SubNoSQLFilterLine("test-320.txt", "sql-train-320.txt")
    LengthNormalize_byte("sql-test.txt", "test-320.txt", 320)
    SubNoSQLFilterLine("test-320.txt", "sql-test-320.txt")
    LengthNormalize_byte("sql-train.txt", "test-360.txt", 360)
    SubNoSQLFilterLine("test-360.txt", "sql-train-360.txt")
    LengthNormalize_byte("sql-test.txt", "test-360.txt", 360)
    SubNoSQLFilterLine("test-360.txt", "sql-test-360.txt")
    LengthNormalize_byte("sql-train.txt", "test-400.txt", 400)
    SubNoSQLFilterLine("test-400.txt", "sql-train-400.txt")
    LengthNormalize_byte("sql-test.txt", "test-400.txt", 400)
    SubNoSQLFilterLine("test-400.txt", "sql-test-400.txt")

    SampleTrainFile("normal-original-noshort-norepeat-urlexchange-FFDD-norepeat-sqlfilter.txt", "normal-ex", 10)
    LengthNormalize_byte("normal-ex-train.txt", "test-80.txt", 80)
    SubNoSQLFilterLine("test-80.txt", "normal-ex-train-80.txt")
    LengthNormalize_byte("normal-ex-test.txt", "test-80.txt", 80)
    SubNoSQLFilterLine("test-80.txt", "normal-ex-test-80.txt")
    LengthNormalize_byte("normal-ex-train.txt", "test-120.txt", 120)
    SubNoSQLFilterLine("test-120.txt", "normal-ex-train-120.txt")
    LengthNormalize_byte("normal-ex-test.txt", "test-120.txt", 120)
    SubNoSQLFilterLine("test-120.txt", "normal-ex-test-120.txt")
    LengthNormalize_byte("normal-ex-train.txt", "test-160.txt", 160)
    SubNoSQLFilterLine("test-160.txt", "normal-ex-train-160.txt")
    LengthNormalize_byte("normal-ex-test.txt", "test-160.txt", 160)
    SubNoSQLFilterLine("test-160.txt", "normal-ex-test-160.txt")
    LengthNormalize_byte("normal-ex-train.txt", "test-200.txt", 200)
    SubNoSQLFilterLine("test-200.txt", "normal-ex-train-200.txt")
    LengthNormalize_byte("normal-ex-test.txt", "test-200.txt", 200)
    SubNoSQLFilterLine("test-200.txt", "normal-ex-test-200.txt")
    LengthNormalize_byte("normal-ex-train.txt", "test-240.txt", 240)
    SubNoSQLFilterLine("test-240.txt", "normal-ex-train-240.txt")
    LengthNormalize_byte("normal-ex-test.txt", "test-240.txt", 240)
    SubNoSQLFilterLine("test-240.txt", "normal-ex-test-240.txt")
    LengthNormalize_byte("normal-ex-train.txt", "test-280.txt", 280)
    SubNoSQLFilterLine("test-280.txt", "normal-ex-train-280.txt")
    LengthNormalize_byte("normal-ex-test.txt", "test-280.txt", 280)
    SubNoSQLFilterLine("test-280.txt", "normal-ex-test-280.txt")
    LengthNormalize_byte("normal-ex-train.txt", "test-320.txt", 320)
    SubNoSQLFilterLine("test-320.txt", "normal-ex-train-320.txt")
    LengthNormalize_byte("normal-ex-test.txt", "test-320.txt", 320)
    SubNoSQLFilterLine("test-320.txt", "normal-ex-test-320.txt")
    LengthNormalize_byte("normal-ex-train.txt", "test-360.txt", 360)
    SubNoSQLFilterLine("test-360.txt", "normal-ex-train-360.txt")
    LengthNormalize_byte("normal-ex-test.txt", "test-360.txt", 360)
    SubNoSQLFilterLine("test-360.txt", "normal-ex-test-360.txt")
    LengthNormalize_byte("normal-ex-train.txt", "test-400.txt", 400)
    SubNoSQLFilterLine("test-400.txt", "normal-ex-train-400.txt")
    LengthNormalize_byte("normal-ex-test.txt", "test-400.txt", 400)
    SubNoSQLFilterLine("test-400.txt", "normal-ex-test-400.txt")

    SampleTrainFile("normal-original-noshort-norepeat-sqlfilter.txt", "normal", 10)
    LengthNormalize_byte("normal-train.txt", "test-80.txt", 80)
    SubNoSQLFilterLine("test-80.txt", "normal-train-80.txt")
    LengthNormalize_byte("normal-test.txt", "test-80.txt", 80)
    SubNoSQLFilterLine("test-80.txt", "normal-test-80.txt")
    LengthNormalize_byte("normal-train.txt", "test-120.txt", 120)
    SubNoSQLFilterLine("test-120.txt", "normal-train-120.txt")
    LengthNormalize_byte("normal-test.txt", "test-120.txt", 120)
    SubNoSQLFilterLine("test-120.txt", "normal-test-120.txt")
    LengthNormalize_byte("normal-train.txt", "test-160.txt", 160)
    SubNoSQLFilterLine("test-160.txt", "normal-train-160.txt")
    LengthNormalize_byte("normal-test.txt", "test-160.txt", 160)
    SubNoSQLFilterLine("test-160.txt", "normal-test-160.txt")
    LengthNormalize_byte("normal-train.txt", "test-200.txt", 200)
    SubNoSQLFilterLine("test-200.txt", "normal-train-200.txt")
    LengthNormalize_byte("normal-test.txt", "test-200.txt", 200)
    SubNoSQLFilterLine("test-200.txt", "normal-test-200.txt")
    LengthNormalize_byte("normal-train.txt", "test-240.txt", 240)
    SubNoSQLFilterLine("test-240.txt", "normal-train-240.txt")
    LengthNormalize_byte("normal-test.txt", "test-240.txt", 240)
    SubNoSQLFilterLine("test-240.txt", "normal-test-240.txt")
    LengthNormalize_byte("normal-train.txt", "test-280.txt", 280)
    SubNoSQLFilterLine("test-280.txt", "normal-train-280.txt")
    LengthNormalize_byte("normal-test.txt", "test-280.txt", 280)
    SubNoSQLFilterLine("test-280.txt", "normal-test-280.txt")
    LengthNormalize_byte("normal-train.txt", "test-320.txt", 320)
    SubNoSQLFilterLine("test-320.txt", "normal-train-320.txt")
    LengthNormalize_byte("normal-test.txt", "test-320.txt", 320)
    SubNoSQLFilterLine("test-320.txt", "normal-test-320.txt")
    LengthNormalize_byte("normal-train.txt", "test-360.txt", 360)
    SubNoSQLFilterLine("test-360.txt", "normal-train-360.txt")
    LengthNormalize_byte("normal-test.txt", "test-360.txt", 360)
    SubNoSQLFilterLine("test-360.txt", "normal-test-360.txt")
    LengthNormalize_byte("normal-train.txt", "test-400.txt", 400)
    SubNoSQLFilterLine("test-400.txt", "normal-train-400.txt")
    LengthNormalize_byte("normal-test.txt", "test-400.txt", 400)
    SubNoSQLFilterLine("test-400.txt", "normal-test-400.txt")

    #LengthFilter("nolength_data\\normal-ex-80.txt", "nolength_data\\normal-ex-80-lenfilter.txt", 80)
    #readByte("test_data\\normal-120-test-lenfilter.txt",120)'''