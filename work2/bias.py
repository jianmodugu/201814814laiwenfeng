import os
import pickle
import numpy as np
from textblob import TextBlob,Word
from nltk.corpus import stopwords
import datetime

#testpath = r'F:\pythonworkspace\MyWork2\Dataset\test'
#trainpath = r'F:\pythonworkspace\MyWork2\Dataset\20news-18828'
nowpath = os.getcwd()
testpath = os.path.join(nowpath,r'Dataset\test')
trainpath = os.path.join(nowpath,r'Dataset\20news-18828')
stop_words = set(stopwords.words('english'))
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
    return listofword,taglist

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

def tag_word(reverselist,taglist):
    tag_word_list = dict()
    tag_word_count = dict()
    
#    print('passageNum =',passage_num)
    for i in range(len(taglist)):
        tag = taglist[i][1]
        tag_word_list[tag] = dict()
        tag_word_count[tag] = 0
    
    for word in reverselist:     
        for passageID in reverselist[word]:
            tag = taglist[passageID][1]
            if not word in tag_word_list[tag]:
                tag_word_list[tag][word] = 0
            tag_word_list[tag][word]+=reverselist[word][passageID]
            tag_word_count[tag]+=reverselist[word][passageID]
            
    return tag_word_list,tag_word_count

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
            if len(reverselist[word])<1 or len(word)/passageNum >0.75:
                removelist.append(word)
        for word in removelist:
            reverselist.pop(word)
        
#        reverselistfile = open(writepath,'wb')
        with open(os.path.join(nowpath,'rl.txt'),'wb') as reverselistfile:  
            pickle.dump(reverselist,reverselistfile)
            reverselistfile.close()
    return reverselist

def pre_treatment(tag_word_list,tag_word_count,reverselist,taglist):
#   每个类的概率
    PV = dict()
    for tag in tag_word_list:
        PV[tag] = 0
    for i in range(len(taglist)):
        tag = taglist[i][1]
        PV[tag]+=1
    for key in PV:
        PV[key] = np.log2(PV[key]/len(taglist))
    
#   P(词|类)
    P = dict()
    for tag in tag_word_list:
        P[tag] = dict()
        for word in reverselist:
            if not word in tag_word_list[tag]:
                P[tag][word] = np.log2(1/len(reverselist))
            else:
#                P[tag][word] = np.log2((tag_word_list[tag][word]*len(taglist)/len(word)+1)/(tag_word_count[tag]+len(reverselist)))
                P[tag][word] = np.log2((tag_word_list[tag][word]+1)/(tag_word_count[tag]+len(reverselist)))
    return PV,P

def excute(PV,P,testpath,number_of_word,taglist):
    count = 0
    listofword,taglist_train = passage2word(testpath,'test')
    score = dict()
    for i in range(len(listofword)):
        for tag in P:
            score[tag] = 0
        for tag in P:
            for word in listofword[i]:
                if not word in P[tag].keys():
                    score[tag] += np.log2(1/number_of_word)
                else:
                    score[tag] += P[tag][word]
            score[tag] += PV[tag]
            
        maxscore = -1000000000000000000#最小值
        maxtag = ''
        for tag in score:
            if score[tag] > maxscore:
                maxscore = score[tag]
                maxtag = tag
        global taglist_test
#        print('passageID:'+str(i))
#        print('predict_tag=',maxtag)
#        print('truetag=',taglist_test[i][1])
        if maxtag == taglist_test[i][1]:
            count+=1
    print('正确率为：')
    print(count/len(taglist_test))

print(datetime.datetime.now())            
#listofword,taglist = passage2word(r'E:\测试文档','train')
listofword,taglist = passage2word(trainpath,'train')
reverselist = wl2rl(listofword)
tag_word_list,tag_word_count = tag_word(reverselist,taglist)
PV,P= pre_treatment(tag_word_list,tag_word_count,reverselist,taglist)
print(datetime.datetime.now())
#excute(PV,P,r'E:\测试文档1',len(reverselist),taglist)
excute(PV,P,testpath,len(reverselist),taglist)
print(datetime.datetime.now())



#for tag in P:
#    print(P[tag])
#    print('这个tag完结')
#print(PV)
#print(tag_word_list)
#print(tag_word_count)
