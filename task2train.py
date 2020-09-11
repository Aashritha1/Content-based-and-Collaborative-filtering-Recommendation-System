import json
import math
import re
import string
import sys
import time
from pyspark import SparkContext
from collections import Counter


def f(x):
    return x


def f1(x):
    items = list()
    for each in x:
        each = re.sub(r'[^\w\s]', '', each)
        each = re.sub("\d+", "", each)
        res = re.split(r"[~\s\r\n]+", each)
        for every in res:
            if every not in stopwords and each != '' and each not in string.ascii_lowercase:
                items.append(every)
    return items


def f2(x):
    items_counter = Counter(x)
    result = []
    items = sorted(items_counter.items(), key=lambda kv: kv[1], reverse=True)
    max_value = items[0][1]
    for key, value in items:
        if value > 3:
            result.append((key, value / max_value))
    return result


def top_200(x):
    result = sorted(x, key=lambda y: -y[1])[:200]
    return list(zip(*result))[0]


def f3(x):
    items = []
    for each in x:
        value = profiles.get(each, '')
        if len(value):
            items.extend(value)
    return list(set(items))


start = time.time()
input_file = sys.argv[1]
stopwords_file = sys.argv[3]
sc = SparkContext("local[*]")
reviews = sc.textFile(input_file).persist().map(lambda x: json.loads(x)).persist()
stopwords = sc.textFile(stopwords_file).map(lambda x: x).collect()
business_ids = reviews.map(lambda x: x["business_id"]).distinct().count()
business_texts_tf = reviews.map(lambda x: (x["business_id"], x["text"].lower())).groupByKey(). \
    map(lambda x: (x[0], f1(x[1]))).map(lambda x: (x[0], f2(x[1]))).flatMapValues(f)
business_texts_idf = business_texts_tf.map(lambda x: (x[1][0], x[0])).groupByKey().map(
    lambda x: (x[0], math.log(business_ids / len(list(x[1])), 2))).collectAsMap()
business_profiles = business_texts_tf.map(lambda x: (x[0], (x[1][0], x[1][1] * business_texts_idf[x[1][0]]))). \
    groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda x: (x[0], top_200(x[1])))
word_map = business_profiles.flatMap(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
profiles = business_profiles.map(lambda x: (x[0], [word_map[each] for each in x[1]])).collectAsMap()
user_profiles = reviews.map(lambda x: (x["user_id"], x["business_id"])).groupByKey().map(
    lambda x: (x[0], list(set(x[1])))).collectAsMap()
res = dict(map(lambda x: (x[0], f3(x[1])), user_profiles.items()))
result = {"user_profiles": res,
          "business_profiles": profiles
          }
with open(sys.argv[2], 'w') as fopen:
    json.dump(result, fopen)
fopen.close()
print("Duration:", time.time() - start)
