import argparse
import math
import os
import string
import warnings

import gensim
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='')
parser.add_argument("--model_path", type=str, default="model/word2vec.model", help="path of the model")
parser.add_argument("--data_path", type=str, default="data/ec_sp500_all.csv", help="path of the data")
parser.add_argument("--threshold", type=float, default=0.5, help="the threshold of counting or not")
parser.add_argument("--vector_size", type=int, default=100, help="how many sizes of your vector, the bigger the more precise")
parser.add_argument("--window", type=int, default=8, help="the size of the window")
parser.add_argument("--topk", type=int, default=10, help="how many companies that you want to know")
parser.add_argument("--min_count", type=int, default=0, help="if smaller, delete")

args = parser.parse_args()


def readFile(file):
    #如此便将所有数据保存在了df里面
    df = pd.read_csv(file, header=0, sep=",", encoding="ISO-8859-1")
    return df

def PreText(df_all):
    #去掉文本里面的\r \n
    df_all["content"] = df_all["content"].replace(r"\r", " ", regex=True)
    df_all["content"] = df_all["content"].replace(r"\n", " ", regex=True)
    #文本里所有的数据大写
    df_all["content"] = df_all["content"].str.upper()
    # 将标点符号都删除
    df_all["content"] = df_all["content"].str.replace(r"\s+", " ", regex=True)
    df_all["content_new"] = df_all["content"].replace(r"[{}]".format(string.punctuation), " ", regex=True).str.split(" ")

    #文本中有多余的空格，比如连续的空格或者行首和行尾的空格，可以去掉，以减少噪音。
    df_all["content"] = df_all["content"].str.split(r"[{}]".format(string.punctuation)).replace(" ", "")

    return df_all

def Train_model(df_Pre_all):
    # df_Pre_all["content"] = df_Pre_all["content"].astype(str)
    #将每一个独立的文本用.进行分割，否则模型会混淆不同的句子。
    # sentences = df_Pre_all["content"].str.cat(sep=".")
    sentences = df_Pre_all["content_new"].to_numpy()
    model_path = args.model_path
    if os.path.exists(model_path):
        model = gensim.models.word2vec.Word2Vec.load(model_path)
    else:
        model = gensim.models.word2vec.Word2Vec(sentences, vector_size = args.vector_size, window = args.window, min_count=args.min_count)
        if not os.path.exists("model") :
            os.makedirs("model")
        model.save(model_path)

    return model

def sumTheWord(df_Pre_all, model):

    first = []
    second = []
    third = []
    fourth = []
    fifth = []

    #先找出词汇表里面大于0.5的相关词汇，
    similar_words_temp = model.wv.most_similar("INNOVATION")

    for word, similarity in similar_words_temp:
        if similarity > args.threshold:
            first.append(word)

    similar_words_temp = model.wv.most_similar("INTEGRITY")
    for word, similarity in similar_words_temp:
        if similarity > args.threshold:
            second.append(word)

    similar_words_temp = model.wv.most_similar("QUALITY")
    for word, similarity in similar_words_temp:
        if similarity > args.threshold:
            third.append(word)

    similar_words_temp = model.wv.most_similar("RESPECT")
    for word, similarity in similar_words_temp:
        if similarity > args.threshold:
            fourth.append(word)

    similar_words_temp = model.wv.most_similar("TEAMWORK")
    for word, similarity in similar_words_temp:
        if similarity > args.threshold:
            fifth.append(word)

    firstDic = pd.DataFrame(columns=first)
    for i in first:
        firstDic[i] = np.zeros(len(df_Pre_all))

    secondDic = pd.DataFrame(columns=second)
    for i in second:
        secondDic[i] = np.zeros(len(df_Pre_all))

    thirdDic = pd.DataFrame(columns=third)
    for i in third:
        thirdDic[i] = np.zeros(len(df_Pre_all))

    fourthDic = pd.DataFrame(columns=fourth)
    for i in fourth:
        fourthDic[i] = np.zeros(len(df_Pre_all))

    fifthDic = pd.DataFrame(columns=fifth)
    for i in fifth:
        fifthDic[i] = np.zeros(len(df_Pre_all))

    for i in range(len(df_Pre_all)):
        for word in first:
            firstDic[word][i] += df_Pre_all["content_new"][i].count(word)
        for word in second:
            secondDic[word][i] += df_Pre_all["content_new"][i].count(word)
        for word in third:
            thirdDic[word][i] += df_Pre_all["content_new"][i].count(word)
        for word in fourth:
            fourthDic[word][i] += df_Pre_all["content_new"][i].count(word)
        for word in fifth:
            fifthDic[word][i] += df_Pre_all["content_new"][i].count(word)

    firstDic["company"] = df_Pre_all["company"]
    secondDic["company"] = df_Pre_all["company"]
    thirdDic["company"] = df_Pre_all["company"]
    fourthDic["company"] = df_Pre_all["company"]
    fifthDic["company"] = df_Pre_all["company"]

    firstDic["fqtr"] = df_Pre_all["fqtr"]
    secondDic["fqtr"] = df_Pre_all["fqtr"]
    thirdDic["fqtr"] = df_Pre_all["fqtr"]
    fourthDic["fqtr"] = df_Pre_all["fqtr"]
    fifthDic["fqtr"] = df_Pre_all["fqtr"]

    return df_Pre_all, firstDic, secondDic,thirdDic,fourthDic, fifthDic
# 定义一个计算某个词的tf-idf值的函数
def tfidf(corpus, word):
    # 词汇表
    dictContainsWords = []
    vocab = set()
    tf = 0
    for doc in corpus:
        while doc.__contains__(""):     doc.remove("")
        cnt = 0
        for temp_word in doc:
            if temp_word == word:
                cnt += 1

        tf += cnt/len(doc)

    vocab = list(vocab)
    # 文档数
    doc_num = len(corpus)
    # 判断词汇是否在词汇表中
    if word not in vocab:
        return 0 # 如果不在，返回0
    else:
        # 计算逆文档频率
        count = 0 # 统计包含该词的文档数

        for doc in corpus:
            if doc.__contains__(word):
                count += 1
        idf = math.log(doc_num / (count + 1)) # 加1是为了避免分母为0
        return tf*idf

def getTFDic(firstDic, contents):
    #contents
    contents = contents.to_numpy()
    for j in range(len(contents)):
        for column in firstDic.columns[0:firstDic.shape[1]-2]:
            firstDic[column][j] = tfidf(contents, column)

    return firstDic

file = args.data_path
df_all = readFile(file)
df_Pre_all = PreText(df_all)
# 将数据扔进模型中进行训练
model = Train_model(df_Pre_all)

# if os.path.exists("results/firstDic.csv"):
if 0:
    firstDic = pd.read_csv("results/firstDic.csv")
    secondDic = pd.read_csv("results/secondDic.csv")
    thirdDic = pd.read_csv("results/thirdDic.csv")
    fourthDic = pd.read_csv("results/fourthDic.csv")
    fifthDic = pd.read_csv("results/fifthDic.csv")
else:
    results, firstDic, secondDic,thirdDic,fourthDic, fifthDic  = sumTheWord(df_Pre_all, model)
    ##计算tf-idf
    firstDic = getTFDic(firstDic, results["content_new"])
    secondDic = getTFDic(secondDic, results["content_new"])
    thirdDic = getTFDic(thirdDic, results["content_new"])
    fourthDic = getTFDic(fourthDic, results["content_new"])
    fifthDic = getTFDic(fifthDic, results["content_new"])

    firstDic["sumFirst"] = firstDic[firstDic.columns[0:firstDic.shape[1]-2]].sum(axis=1)
    secondDic["sumSecond"] = secondDic[secondDic.columns[0:secondDic.shape[1]-2]].sum(axis=1)
    thirdDic["sumThird"] = thirdDic[thirdDic.columns[0:thirdDic.shape[1]-2]].sum(axis=1)
    fourthDic["sumFourth"] = fourthDic[fourthDic.columns[0:fourthDic.shape[1]-2]].sum(axis=1)
    fifthDic["sumFifth"] = fifthDic[fifthDic.columns[0:fifthDic.shape[1]-2]].sum(axis=1)

    firstDic.to_csv("results/firstDic.csv")
    secondDic.to_csv("results/secondDic.csv")
    thirdDic.to_csv("results/thirdDic.csv")
    fourthDic.to_csv("results/fourthDic.csv")
    fifthDic.to_csv("results/fifthDic.csv")
results = pd.concat([df_Pre_all["company"], df_Pre_all["fqtr"], df_Pre_all["fyr"], firstDic["sumFirst"], secondDic["sumSecond"], thirdDic["sumThird"], fourthDic["sumFourth"], fifthDic["sumFifth"]], axis=1,
                    keys = ["company", "fqtr", "fyr", "Innovation", "Integrity", "Quality", "Respect", "Teamwork"])
Allyears = results["fyr"].drop_duplicates().to_numpy()
for year in Allyears:
    resultsTemp = results[results["fyr"] == year]
    resultsTemp.to_csv("results/AllScore" + str(year) + ".csv")
results.to_csv("results/AllScore.csv")