import re
from collections import defaultdict

import jieba
from bs4 import BeautifulSoup

stopwords = ['一', '一下', '一些', '一切', '一则', '一天', '一定', '一方面', '一旦', '一时', '一来', '一样', '一次', '一片', '一致', '一般', '一起', '一边',
             '一面', '万一', '上下', '上升', '上去', '上来', '上述', '上面', '下列', '下去', '下来', '下面', '不一', '不久', '不仅', '不会', '不但', '不光',
             '不单', '不变', '不只', '不可', '不同', '不够', '不如', '不得', '不怕', '不惟', '不成', '不拘', '不敢', '不断', '不是', '不比', '不然', '不特',
             '不独', '不管', '不能', '不要', '不论', '不足', '不过', '不问', '与', '与其', '与否', '与此同时', '专门', '且', '两者', '严格', '严重', '个',
             '个人', '个别', '中小', '中间', '乃', '乃至', '么', '之', '之一', '之前', '之后', '之後', '之所以', '之类', '乌乎', '乎', '乘', '也',
             '今后', '今天', '今年', '吧', '吧哒', '吱', '呀', '呃', '呕', '呗', '呜', '呜呼', '呢', '周围', '呵', '呸', '呼哧', '咋', '和', '咚',
             '咦', '咱', '咱们']


def review_to_wordlist(review, remove_stopwords=True):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review, "html.parser").get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[\s+\d+\.\!\/_,$%^*(+\"\']+|[+——，。？、~@#￥%……&*（）“”]+", " ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = jieba.cut(review_text)
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        words = [w for w in words if not w in stopwords]
    #
    # 5. Return a list of words
    return words

def get_list_words(data):
    words_list = []
    for sentence in data:
        words = review_to_wordlist(sentence, remove_stopwords=True)
        words_list.append(words)

    return  words_list

def get_volcabulary_and_list_words(data):
    reviews_words = []
    volcabulary = defaultdict(int)
    for review in data:
        review_words = review_to_wordlist(review, remove_stopwords=True)
        reviews_words.append(review_words)
        for word in review_words:
            volcabulary[word] += 1
        del review_words
    return volcabulary, reviews_words


def load_corpus():
    text = []
    labels = []
    with open("data/neg.txt", encoding='utf-8') as neg_file:
        lines = neg_file.readlines()

        for line in lines:
            text.append(line)
            labels.append(0)

    with open("data/pos.txt", encoding='utf-8') as neg_file:
        lines = neg_file.readlines()

        for line in lines:
            text.append(line)
            labels.append(1)
    return text, labels

# if __name__ == "__main__":
