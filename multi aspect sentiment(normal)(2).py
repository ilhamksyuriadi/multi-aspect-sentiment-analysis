# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:02:34 2019

@author: Asus
"""
import xlrd
import re
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from math import log10
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

def LoadDataset(FileLoc):#load dataset
    print("Load Dataset")
    data = []
    availability = []
    accessability = []
    information = []
    time = []
    customer_service = []
    comfort = []
    safety = []
    workbook = xlrd.open_workbook(FileLoc)
    sheet = workbook.sheet_by_index(0)
    for i in range(1,sheet.nrows):
        data.append(sheet.cell_value(i,0))
        availability.append(sheet.cell_value(i,1))
        accessability.append(sheet.cell_value(i,2))
        information.append(sheet.cell_value(i,3))
        time.append(sheet.cell_value(i,4))
        customer_service.append(sheet.cell_value(i,5))
        comfort.append(sheet.cell_value(i,6))
        safety.append(sheet.cell_value(i,7))
    return data, availability, accessability, information, time, customer_service, comfort, safety

def LoadKataFormal(FileLoc):#Load data normal
    print("Load Kata Normal")
    gakFormal = []
    formal = []
    workbook = xlrd.open_workbook(FileLoc)
    sheet = workbook.sheet_by_index(0)
    for i in range(1,sheet.nrows):
        gakFormal.append(sheet.cell_value(i,0))
        formal.append(sheet.cell_value(i,1))
    return gakFormal, formal

def Preprocessing(data,gakFormal,formal):
    print("Preprocessing")
    cleanData = []
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = StemmerFactory().create_stemmer()
    count = 0
    for kalimat in data:
        removedHttp = re.sub(r"http\S+", '', kalimat)#hilangin link http
        removedPic = re.sub(r"pic.twitter\S+", '', removedHttp)#hilangin link pic.twitter
        lower = removedPic.lower()#casefolding
        tokenized = tokenizer.tokenize(lower)#tokenizer + punctuation removal
        stemmed = []#stemming
        for kata in tokenized:#stemming
            stemmed.append(stemmer.stem(kata))#stemming
        kataFormal = []#ubah kata jadi formal
        for i in range(len(stemmed)):#ubah kata jadi formal
            if stemmed[i] in gakFormal:
                kataFormal.append(formal[gakFormal.index(stemmed[i])])
            else:
                kataFormal.append(stemmed[i])
        cleanData.append(kataFormal)
        count += 1
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

def BagiData(docU,docB,label):
    newDocU = []
    newDocB = []
    newLabel = []
    for i in range(len(label)):
        if label[i] == -1.0 or label[i] == 1.0:
            newDocU.append(docU[i])
            newDocB.append(docB[i])
            newLabel.append(label[i])
    return newDocU,newDocB,newLabel

def ConfusionMatrics(label,predict):
    print("Count Confusion Matrics")
    tn, fp, fn, tp = confusion_matrix(label,predict).ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    return acc, tn, fp, fn, tp

rawData, rawAvailability, rawAccessability, rawInformation, rawTime, rawCustomer_service, rawComfort, rawSafety = LoadDataset('labeling rev. 2.xlsx')
gakFormal,formal = LoadKataFormal('normalisasi.xlsx')
cleanData = Preprocessing(rawData,gakFormal,formal)
cleanDataBigram = DataBigram(cleanData)

unigram = CreateUnigram(cleanData)
bigram = CreateBigram(cleanData)

unigramDf = CreateDf(unigram,cleanData)
bigramDf = CreateDf(bigram,cleanDataBigram)

unigramTfidf = CreateTfidf(cleanData,unigramDf,unigram)
bigramTfidf = CreateTfidf(cleanDataBigram,bigramDf,bigram)

availabilityDocU,availabilityDocB,availabilityLabel = BagiData(unigramTfidf,bigramTfidf,rawAvailability)
accessabilityDocU,accessabilityDocB,accessabilityLabel = BagiData(unigramTfidf,bigramTfidf,rawAccessability)
informationDocU,informationDocB,informationLabel = BagiData(unigramTfidf,bigramTfidf,rawInformation)
timeDocU,timeDocB,timeLabel = BagiData(unigramTfidf,bigramTfidf,rawTime)
customer_serviceDocU,customer_serviceDocB,customer_serviceLabel = BagiData(unigramTfidf,bigramTfidf,rawCustomer_service)
comfortDocU,comfortDocB,comfortLabel = BagiData(unigramTfidf,bigramTfidf,rawComfort)
safetyDocU,safetyDocB,safetyLabel = BagiData(unigramTfidf,bigramTfidf,rawSafety)

clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

predictAvailabilityU = cross_val_predict(clf, availabilityDocU, availabilityLabel, cv=10)
predictAvailabilityU.tolist()
predictAvailabilityB = cross_val_predict(clf, availabilityDocB, availabilityLabel, cv=10)
predictAvailabilityB.tolist()

predictAccessabilityU = cross_val_predict(clf, accessabilityDocU, accessabilityLabel, cv=10)
predictAccessabilityU.tolist()
predictAccessabilityB = cross_val_predict(clf, accessabilityDocB, accessabilityLabel, cv=10)
predictAccessabilityB.tolist()

predictInformationU = cross_val_predict(clf, informationDocU, informationLabel, cv=10)
predictInformationU.tolist()
predictInformationB = cross_val_predict(clf, informationDocB, informationLabel, cv=10)
predictInformationB.tolist()

predictTimeU = cross_val_predict(clf, timeDocU, timeLabel, cv=10)
predictTimeU.tolist()
predictTimeB = cross_val_predict(clf, timeDocB, timeLabel, cv=10)
predictTimeB.tolist()

predictCustomer_serviceU = cross_val_predict(clf, customer_serviceDocU, customer_serviceLabel, cv=10)
predictCustomer_serviceU.tolist()
predictCustomer_serviceB = cross_val_predict(clf, customer_serviceDocB, customer_serviceLabel, cv=10)
predictCustomer_serviceB.tolist()

predictComfortU = cross_val_predict(clf, comfortDocU, comfortLabel, cv=10)
predictComfortU.tolist()
predictComfortB = cross_val_predict(clf, comfortDocB, comfortLabel, cv=10)
predictComfortB.tolist()

predictSafetyU = cross_val_predict(clf, safetyDocU, safetyLabel, cv=10)
predictSafetyU.tolist()
predictSafetyB = cross_val_predict(clf, safetyDocB, safetyLabel, cv=10)
predictSafetyB.tolist()

predictAvailabilityU = cross_val_predict(clf, availabilityDocU, availabilityLabel, cv=10)
predictAvailabilityU.tolist()
predictAvailabilityB = cross_val_predict(clf, availabilityDocB, availabilityLabel, cv=10)
predictAvailabilityB.tolist()

predictAccessabilityU = cross_val_predict(clf, accessabilityDocU, accessabilityLabel, cv=10)
predictAccessabilityU.tolist()
predictAccessabilityB = cross_val_predict(clf, accessabilityDocB, accessabilityLabel, cv=10)
predictAccessabilityB.tolist()

predictInformationU = cross_val_predict(clf, informationDocU, informationLabel, cv=10)
predictInformationU.tolist()
predictInformationB = cross_val_predict(clf, informationDocB, informationLabel, cv=10)
predictInformationB.tolist()

predictTimeU = cross_val_predict(clf, timeDocU, timeLabel, cv=10)
predictTimeU.tolist()
predictTimeB = cross_val_predict(clf, timeDocB, timeLabel, cv=10)
predictTimeB.tolist()

predictCustomer_serviceU = cross_val_predict(clf, customer_serviceDocU, customer_serviceLabel, cv=10)
predictCustomer_serviceU.tolist()
predictCustomer_serviceB = cross_val_predict(clf, customer_serviceDocB, customer_serviceLabel, cv=10)
predictCustomer_serviceB.tolist()

predictComfortU = cross_val_predict(clf, comfortDocU, comfortLabel, cv=10)
predictComfortU.tolist()
predictComfortB = cross_val_predict(clf, comfortDocB, comfortLabel, cv=10)
predictComfortB.tolist()

predictSafetyU = cross_val_predict(clf, safetyDocU, safetyLabel, cv=10)
predictSafetyU.tolist()
predictSafetyB = cross_val_predict(clf, safetyDocB, safetyLabel, cv=10)
predictSafetyB.tolist()

accAvailabilityU = ConfusionMatrics(predictAvailabilityU,availabilityLabel)
accAvailabilityB = ConfusionMatrics(predictAvailabilityB,availabilityLabel)

accAccessabilityU = ConfusionMatrics(predictAccessabilityU,accessabilityLabel)
accAccessabilityB = ConfusionMatrics(predictAccessabilityB,accessabilityLabel)

accInformationU = ConfusionMatrics(predictInformationU,informationLabel)
accInformationB = ConfusionMatrics(predictInformationB,informationLabel)

accTimeU = ConfusionMatrics(predictTimeU,timeLabel)
accTimeB = ConfusionMatrics(predictTimeB,timeLabel)

accCustomer_serviceU = ConfusionMatrics(predictCustomer_serviceU,customer_serviceLabel)
accCustomer_serviceB = ConfusionMatrics(predictCustomer_serviceB,customer_serviceLabel)

accComfortU = ConfusionMatrics(predictComfortU,comfortLabel)
accComfortB = ConfusionMatrics(predictComfortB,comfortLabel)

accSafetyU = ConfusionMatrics(predictSafetyU,safetyLabel)
accSafetyB = ConfusionMatrics(predictSafetyB,safetyLabel)




