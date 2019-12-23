# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:02:34 2019

@author: Asus
"""
import xlrd
import re
from nltk.tokenize import RegexpTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from math import log10
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

def LoadDataset(FileLoc):#load dataset
    print("Load Dataset")
    data = []
    available = []
    access = []
    info = []
    time = []
    service = []
    comfort = []
    safety = []
    workbook = xlrd.open_workbook(FileLoc)
    sheet = workbook.sheet_by_index(0)
    for i in range(1,sheet.nrows):
        data.append(sheet.cell_value(i,0))
        available.append(sheet.cell_value(i,1))
        access.append(sheet.cell_value(i,2))
        info.append(sheet.cell_value(i,3))
        time.append(sheet.cell_value(i,3))
        service.append(sheet.cell_value(i,3))
        comfort.append(sheet.cell_value(i,3))
        safety.append(sheet.cell_value(i,3))
    return data, available, access, info, time, service, comfort, safety

def Preprocessing(data):
    print("Preprocessing")
    cleanData = []
    tokenizer = RegexpTokenizer(r'\w+')
    factory_stopwords = StopWordRemoverFactory()
    stopwordsFact = factory_stopwords.get_stop_words()
    stemmer = StemmerFactory().create_stemmer()
    count = 0
    for kalimat in data:
        removedHttp = re.sub(r"http\S+", '', kalimat)#hilangin link http
        removedPic = re.sub(r"pic.twitter\S+", '', removedHttp)#hilangin link pic.twitter
        lower = removedPic.lower()#casefolding
        tokenized = tokenizer.tokenize(lower)#tokenizer + punctuation removal
        stopwords = []#Stopwords removal
        for kata in tokenized:
            if kata not in stopwordsFact:
                stopwords.append(kata)
        stemmed = []#stemming
        for kata in stopwords:#stemming
            stemmed.append(stemmer.stem(kata))#stemming
        cleanData.append(stemmed)
        count += 1
        print(count)
    return cleanData

def DataBigram(doc):
    print("Create with bigram form")
    newDoc = []
    for kalimat in doc:
        tempKalimat = []
        for i in range(len(kalimat)-1):
            bigram = kalimat[i] + ' ' + kalimat[i+1]
            tempKalimat.append(bigram)
        newDoc.append(tempKalimat)
    return newDoc

def CreateUnigram(data):
    print("Create Unigram")
    unigram = []
    for kalimat in data:
        for kata in kalimat:
            if kata not in unigram:
                unigram.append(kata)
    return unigram

def CreateBigram(data):
    print("Create Bigram")
    bigram = []
    for kalimat in data:
        for i in range(len(kalimat)-1):
            tempBigram = kalimat[i] + ' ' + kalimat[i+1]
            if tempBigram not in bigram:
                bigram.append(tempBigram)
    return bigram

def CreateDf(data,doc):
    print("Count DF")
    df = {}
    for kata in data:
        for kalimat in doc:
            if kata in kalimat:
                if kata not in df:
                    df[kata] = 1
                else:
                    df[kata] += 1
    return df

def CreateTfidf(data,df,term):
    print("Count TFIDF")
    tfidf = []
    count = 0
    for i in range(len(data)):
        tempTfidf = []
        for j in range(len(term)):
            if term[j] in data[i]:
                tf = 0
                for k in range(len(data[i])):
                    if data[i][k] == term[j]:
                        tf += 1
                idf = log10(len(data)/df[term[j]])
                tempTfidf.append(idf*tf)
            else:
                tempTfidf.append(0)
        count += 1
        tfidf.append(tempTfidf)
    return tfidf

def BagiData(docU,docB,docUB,label):
    newDocU = []
    newDocB = []
    newDocUB = []
    newLabel = []
    for i in range(len(label)):
        if label[i] == -1.0 or label[i] == 1.0:
            newDocU.append(docU[i])
            newDocB.append(docB[i])
            newDocUB.append(docUB[i])
            newLabel.append(label[i])
    return newDocU,newDocB,newDocUB,newLabel

def FeatureMerger(data1,data2):
    print("Kombinasi Unigram Bigram")
    mergedData = []
    for i in range(len(data1)):
        value = data1[i] + data2[i]
        mergedData.append(value)
    return mergedData

def ConfusionMatrics(label,predict):
    print("Count Confusion Matrics")
    tn, fp, fn, tp = confusion_matrix(label,predict).ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    return acc

rawData, rawAvailable, rawAccess, rawInfo, rawTime, rawService, rawComfort, rawSafety = LoadDataset('labeling rev.2(2).xlsx')
cleanData = Preprocessing(rawData)
cleanDataBigram = DataBigram(cleanData)

unigram = CreateUnigram(cleanData)
bigram = CreateBigram(cleanData)

unigramDf = CreateDf(unigram,cleanData)
bigramDf = CreateDf(bigram,cleanDataBigram)

unigramTfidf = CreateTfidf(cleanData,unigramDf,unigram)
bigramTfidf = CreateTfidf(cleanDataBigram,bigramDf,bigram)
unibiTfidf = FeatureMerger(unigramTfidf,bigramTfidf)

availableDocU,availableDocB,availableDocUB,availableLabel = BagiData(unigramTfidf,bigramTfidf,unibiTfidf,rawAvailable)
accessDocU,accessDocB,accessDocUB,accessLabel = BagiData(unigramTfidf,bigramTfidf,unibiTfidf,rawAccess)
infoDocU,infoDocB,infoDocUB,infoLabel = BagiData(unigramTfidf,bigramTfidf,unibiTfidf,rawInfo)
timeDocU,timeDocB,timeDocUB,timeLabel = BagiData(unigramTfidf,bigramTfidf,unibiTfidf,rawTime)
serviceDocU,serviceDocB,serviceDocUB,serviceLabel = BagiData(unigramTfidf,bigramTfidf,unibiTfidf,rawService)
comfortDocU,comfortDocB,comfortDocUB,comfortLabel = BagiData(unigramTfidf,bigramTfidf,unibiTfidf,rawComfort)
safetyDocU,safetyDocB,safetyDocUB,safetyLabel = BagiData(unigramTfidf,bigramTfidf,unibiTfidf,rawSafety)

clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

#Klasifikasi Availability
#predictAvailableU = cross_val_predict(clf, availableDocU, availableLabel, cv=2)
#predictAvailableU.tolist()
#predictAvailableB = cross_val_predict(clf, availableDocB, availableLabel, cv=2)
#predictAvailableB.tolist()
#predictAvailableUB = cross_val_predict(clf, availableDocUB, availableLabel, cv=2)
#predictAvailableUB.tolist()

#Klasifikasi Access
predictAccessU = cross_val_predict(clf, accessDocU, accessLabel, cv=2)
predictAccessU.tolist()
predictAccessB = cross_val_predict(clf, accessDocB, accessLabel, cv=2)
predictAccessB.tolist()
predictAccessUB = cross_val_predict(clf, accessDocUB, accessLabel, cv=2)
predictAccessUB.tolist()

#Info (ikutin yang atas)


#Akurasi Availability
#accAvailableU = ConfusionMatrics(predictAvailableU,availableLabel)
#accAvailableB = ConfusionMatrics(predictAvailableB,availableLabel)
#accAvailableUB = ConfusionMatrics(predictAvailableUB,availableLabel)

#Akurasi Accesss
accAccessU = ConfusionMatrics(predictAccessU,accessLabel)
accAccessB = ConfusionMatrics(predictAccessB,accessLabel)
accAccessUB = ConfusionMatrics(predictAccessUB,accessLabel)

#Akurasi Info (ikutin yang atas)



#Akurasi rata-rata
#rataU = (accAvailableU + accAccessU) / 2 #kalau udah tambahin sesuai akurasi aspek yg lainnya
#rataB = (accAvailableB + accAccessB) / 2 #kalau udah tambahin sesuai akurasi aspek yg lainnya
#rataUB = (accAvailableUB + accAccessUB) / 2 #kalau udah tambahin sesuai akurasi aspek yg lainnya











