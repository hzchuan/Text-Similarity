#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 12:26:27 2018

@author: hzchuan
"""
import math
import numpy as np
import re
import pandas as pd
import time
#讀資料
def loadData(file):
    with open(file,'r',encoding='gbk') as fp:
        line = fp.readlines()
#    line = line[0:4000] #***小資料測試
    temp = []
    for i in range(len(line)):#切分篇章
        if(len(line[i])==1 and line[i]=='\n'):
            temp.append(i)
            if(temp.index(i)!=0):
                for x in range(temp[temp.index(i)-1]+2,i):
                    line[temp[temp.index(i)-1]+1] += line[x]
            if(temp.index(i)==0):
                for x in range(1,i):
                    line[0]+=line[x]
    data = []
    data.append(line[0])
    for x in temp:
        try:
            data.append(line[x+1])
        except: pass               
    return data

def cleanData(data):
    r1 = u'[a-zA-Z0-9’"#$%&\'()（）《》：．*+,-./:;，。、…？“”‘’！[\\]^_`{|}~]+'
    data = [re.sub(r1,'',x) for x in data]
    temp_data = [] #使用正則表達式 清理詞性標註
    for x in data:
        temp = x.replace('\t', ' ').strip('\r\n').split(' ')
        temp_data.append(temp)
    for i in range(len(temp_data)): #清除空格 
        for x in temp_data[i]:
            temp_data[i] = [x for x in temp_data[i] if x!='' and x!='\n']   
    
    stopwords = [] #讀取停用詞庫
    with open(stopword_file, 'r', encoding='utf-8') as fp:
        for x in fp:
            stopwords.append(x.strip())
    fp.close()
    ndf= [] #去除停用詞
    for x in temp_data:
        temp = []
        for m in x:
            if m not in stopwords:
                temp.append(m) 
        ndf.append(temp)
    ndf = [x for x in ndf if len(x)!= 0] #清理空行
    return ndf

#建立詞袋
def getWordbag(ndf):
    wordCount= {}
    lineArr = [] #依序出現的所有詞彙
    for i in range(len(ndf)):
        for x in ndf[i]:
            lineArr.append(x)
    for i in range(1, len(lineArr)):#詞袋建立，計算詞彙出現總次數
        if lineArr[i] not in wordCount:
            wordCount[lineArr[i]] = 1
        else: wordCount[lineArr[i]] += 1
    for word in list(wordCount.keys()):#刪除只出現過一次的詞
        if wordCount[word] <= 10:
            del wordCount[word]
    wordsbag = []#詞袋向量
    for i in wordCount:
        wordsbag.append(i)
    return wordsbag
     
def cal_tfidf(ndf):
    tf = [] #各篇章內詞彙出現次數
    st1 = time.time()
    for line in ndf:
        tf_dic = {}
        for index in range(0, len(line)):  # Tf值计算
            if line[index] not in tf_dic:
                tf_dic[line[index]] = 1
            else:
                tf_dic[line[index]] += 1
        tf.append(tf_dic)    
    tf_list = []#各篇章內詞彙出現比例
    for i in tf:  # tf 公式：tf = (0.5 + 0.5*(tf/maxTf))*(1/len(w))
        sort_dic = sorted(i.items(), key=lambda d: d[1], reverse=True)  # 关键词重要性排序
        temp_dic = {}
        for j in range(0, len(sort_dic)):
            max_tf = sort_dic[0][1]
            temp_dic[sort_dic[j][0]] = np.dot((0.5 + 0.5*(sort_dic[j][1]/max_tf)),(1.0/len(sort_dic)))
        tf_list.append(temp_dic)  
        
    idf = [] ##Inverse Data Frequency (idf)各篇章內詞彙在所有文本中出現次數
    for line in tf_list:
        temp_dic = {}
        for word in line:
            for check_line in tf_list:
                if word in check_line:
                    if word not in temp_dic:
                        temp_dic[word] = 1
                    else:
                        temp_dic[word] += 1
        idf.append(temp_dic)
    file_len = len(idf)    
    tf_idf = [] #idf = ln(N/n), tf-idf = idf*tf
    for line in range(0, len(idf)):
        temp_dic = {}
        for word in idf[line]:
            temp_dic[word] = np.dot(math.log((file_len + 1)/int(idf[line][word])),tf_list[line][word])
        tf_idf.append(temp_dic)
    et1 = time.time()
    print('tf-idf計算時間：'+str(et1- st1)+' 秒')
    return tf_idf

#計算餘弦值
def cal_cos(tf_idf):
    st2 = time.time()
#    wordsbag = getWordbag(ndf)
    all_vec = []  
    for j in range(len(tf_idf)):
        text_vec = []
        for i in range(len(wordsbag)):
            text_vec.append(0)       
        for k in tf_idf[j]:
            if k in wordsbag:
                text_vec[wordsbag.index(k)] += tf_idf[j][k]
        all_vec.append(text_vec)    
    all_vec = np.array(all_vec)

    mtx = np.zeros([len(all_vec), len(all_vec)])
    length =[]
    for i in range(len(all_vec)):
        length.append(np.linalg.norm(all_vec[i]))
    
    for i in range(len(all_vec)):
        for j in range(i, len(all_vec)):
            if(i==j):
                mtx[i][j] = 1            
            else:
                mtx[i][j] = np.dot(all_vec[i],all_vec[j])/(length[i]*length[j])
                mtx[j][i] = mtx[i][j]
        print(i)  
    et2 = time.time()
    print('餘弦值計算時間：',str(et2- st2)+' 秒')
    return mtx
    
if __name__ == '__main__': 
    file = '199801.txt'
    stopword_file = 'stopword.txt'
    ndf = cleanData(loadData(file))
    wordsbag = getWordbag(ndf)
    tf_idf = cal_tfidf(ndf) 
    mtx = cal_cos(tf_idf)


    