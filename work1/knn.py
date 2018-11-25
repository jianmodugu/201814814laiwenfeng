# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from textblob import TextBlob,Word
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
import datetime
from collections import Counter

stop_words = set(stopwords.words('english'))
trainpath = r'F:\pythonworkspace\KNN\dataset\20news-18828'
testpath = r'F:\pythonworkspace\KNN\dataset\test'
nowpath = os.getcwd()
global passageNum

#存储测试集信息
global taglist_test

def passage2word(path=trainpath,type = 'train'):
    listofword = []
    filelist=[]
    taglist=[]
    traverse(path,filelist,taglist)
    
    if(type=='train'):
#       id_tag = open(r'E:\id_tag.txt','wb')
        
        with open(os.path.join(nowpath,'id_tag.txt'),'wb') as id_tag:
            pickle.dump(taglist,id_tag)
            id_tag.close()
    elif(type=='test'):
        global taglist_test
        taglist_test = taglist
    
    global passageNum
    passageNum = len(filelist)     
    for file in filelist:
        f = open(file,'r',errors='ignore')
        passage = f.read()
        f.close()
        str = TextBlob(passage.strip())
        
        wordlist =str.lower().words
        k = len(wordlist)
        answer = []
        if k > 0:
            for x in range(k):
                answer.append(wordlist[x].lemmatize())
                answer[x] = Word(answer[x]).lemmatize("v")  
        filtered_answer = [x for x in answer if not x in stop_words]
#        print(filtered_answer)
        listofword.append(filtered_answer)
    return listofword

#
def testpassage2vector(path):
    listofword = passage2word(path,'test')
    print(listofword)
    test_Num = len(listofword)
    testreverselist = wl2rl(listofword,'no')
    print(testreverselist,'这行')
    
    #维度映射文件
    drfile = open(os.path.join(nowpath,'dr.txt'),'rb')
#    with open(os.path.join(nowpath,'dr.txt'),'rb',) as drfile:
    dimension_record = pickle.load(drfile)
    drfile.close()
    
    #词典文件
#    rlfile = open(r'E:\rl.txt','rb')
    rlfile = open(os.path.join(nowpath,'rl.txt'),'rb')
    reverselist = pickle.load(rlfile)
    rlfile.close()
    
    testvsm = []
    print(len(reverselist))
    for i in range(test_Num):
        testvsm.append([0]*len(reverselist))
    
    for word in testreverselist:
        if not word in dimension_record:
            continue
        dimension_Num = dimension_record[word]
        for passageID in testreverselist[word]:
            if not passageID =='idf':
                testvsm[passageID][dimension_Num]=testreverselist[word][passageID]*(np.log(reverselist[word]['idf']))
            
    return testvsm

def wl2rl(listofword,writepath = r'E:\rl.txt'):
    reverselist = dict()
    passage_num = len(listofword)
    print('passageNum =',passage_num)
    for i in range(passage_num):
        passage = listofword[i]
        for word in passage:
            if not(word in reverselist):
                reverselist[word] = dict()
                reverselist[word][i] = 1
            else:
#                print('word='+word+' i='+str(i))
                if not(i in reverselist[word]):
                    reverselist[word][i] = 0   #没有就初始化一个
                reverselist[word][i] = reverselist[word][i] + 1
    
    #no表示是测试集合
    if not writepath=='no':
        removelist = []  
        for word in reverselist:
            if len(reverselist[word])<5 or len(word)/passageNum >0.75:
                removelist.append(word)
            else:
                reverselist[word]['idf'] = passageNum/len(word)
        for word in removelist:
            reverselist.pop(word)
        
#        reverselistfile = open(writepath,'wb')
        with open(os.path.join(nowpath,'rl.txt'),'wb') as reverselistfile:  
            pickle.dump(reverselist,reverselistfile)
            reverselistfile.close()
    return reverselist

def rl2vsm(reverselist):
    dimension = len(reverselist)
    vector = []
    global passageNum
    for i in range(passageNum):
        vector.append([0]*dimension)
    
    count = 0  #指示当前处理到了第几维
    word_dimension = dict()
    for word in reverselist.keys():
        for passageID in reverselist[word].keys():
            if not passageID =='idf':
                vector[int(passageID)][count] = reverselist[word][passageID]*(np.log(reverselist[word]['idf']))
        word_dimension[word] = count
        count = count+1
#    print('vector =',vector)
    
#   vectorfile = open(r'E:\vsm.txt','wb')
    with open(os.path.join(nowpath,'vsm.txt'),'wb') as vectorfile:
        pickle.dump(vector,vectorfile)
        vectorfile.close()
    
#    dimension_record = open(r'E:\dr.txt','wb')
    with open(os.path.join(nowpath,'dr.txt'),'wb') as dimension_record:
        pickle.dump(word_dimension,dimension_record)
        dimension_record.close()
    
    return vector
        

def knn(vectorspace,vector_test,k):

    train = csr_matrix(np.mat(vectorspace).T)
    test = csr_matrix(np.mat(vector_test))
    
    #计算A·B
    AB = (test)*(train)
    AB = AB/np.sqrt((test.multiply(test).sum(axis=1)))
    AB = AB/np.sqrt((train.multiply(train).sum(axis=0)))
    
    C = np.argsort(AB)
    for x in range(k):
        vote(C[:,-x-1:])
    
     
def vote(list1):
    list = list1.tolist()
    count = 0
    global taglist_test
    taglist = taglist_test
    
    f = open(os.path.join(nowpath,'id_tag.txt'),'rb')
    id_tag = pickle.load(f)
    f.close()
    
    for i in range(len(list)):
        tags = []
        for x in list[i]:
            tags.append(id_tag[x][1])
        c = Counter(tags)
        maxtag = c.most_common(1)
#        print(taglist[i][1],' ',maxtag[0][0])
        if taglist[i][1] == maxtag[0][0]:
            count+=1
    print('k=',len(list[0]),'时，正确率为：',count/len(list))
    
def excute(path,k):
#    vectorfile = open(r'E:\vsm.txt','rb')
    vectorfile = open(os.path.join(nowpath,'vsm.txt'),'rb')
    vectorspace = pickle.load(vectorfile)
    vectorfile.close()

    testvector = testpassage2vector(path)
    print(len(vectorspace[0]),len(testvector[0]))
    print('start knn:',datetime.datetime.now())
    knn(vectorspace,testvector,k)
    print('end knn:',datetime.datetime.now())
     
def generateVSM(path = trainpath):
    rl2vsm(wl2rl(passage2word(path)))
    

        
def traverse(f,filelist,taglist):

    fs = os.listdir(f)
    for f1 in fs:
        tmp_path = os.path.join(f,f1)
        if not os.path.isdir(tmp_path):
            filelist.append(tmp_path)

            passage = []
            passage.append(tmp_path.split('\\')[-1])
            passage.append(tmp_path.split('\\')[-2])
            taglist.append(passage)
        else:
            traverse(tmp_path,filelist,taglist)
            
    
        


generateVSM()
excute(testpath,50)

#traverse(trainpath,[])

