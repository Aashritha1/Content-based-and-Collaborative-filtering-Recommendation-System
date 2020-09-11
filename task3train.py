import json
import math
import random
import sys
import time
from itertools import combinations
from pyspark import SparkContext

p = 99999999977
a = random.sample(range(1, p), 40)
b = random.sample(range(0, p), 40)


def hash_ids(x):
    hashed = []
    for i in range(40):
        hashed.append((((a[i] * x) + b[i]) % p) % m)
    return hashed


def f(x):
    res = {}
    for each in x:
        res[each[0]] = each[1]
    return res


def f1(x):
    return x


def f2(x):
    partition = []
    for i, each in enumerate(range(0, Bands, rows)):
        partition.append((i, tuple(x[each:each + rows])))
    return partition


def similarity(x, y):
    common_users = set(x.keys()) & set(y.keys())
    l = len(common_users)
    numerator_res = 0
    x_users, y_users = [], []
    for each in common_users:
        x_users.append(x[each])
        y_users.append(y[each])
    x_avg = sum(x_users) / l
    y_avg = sum(y_users) / l
    try:
        x_data = 0
        y_data = 0
        for each in range(len(common_users)):
            numerator_res += ((x_users[each] - x_avg) * (y_users[each] - y_avg))
        for each in range(len(common_users)):
            x_data += ((x_users[each] - x_avg) ** 2)
            y_data += ((y_users[each] - y_avg) ** 2)
        denominator_res = math.sqrt(x_data) * math.sqrt(y_data)
        return numerator_res / denominator_res
    except:
        return 0


def jaccard(x):
    try:
        column1 = set(user_business_map[x[0]].keys())
        column2 = set(user_business_map[x[1]].keys())
        if len(column1 & column2) >= 3:
            if (len(column1 & column2) / len(column1 | column2)) >= 0.01:
                return True
    except:
        return False


def check_n_neighbors(x, y):
    if x and y:
        if len(set(x.keys()) & set(y.keys())) >= 3:
            return True
    return False


start = time.time()
input_file = sys.argv[1]
cf_type = sys.argv[3]
sc = SparkContext("local[*]")
if cf_type == "item_based":
    reviews = sc.textFile(input_file).persist().map(lambda x: json.loads(x)).map(
        lambda x: (x["business_id"], x["user_id"], x["stars"]))
    business_users_map = reviews.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().map(
        lambda x: (x[0], list(set(x[1])))).filter(lambda x: len(x[1]) >= 3).map(
        lambda x: (x[0], f(x[1]))).collectAsMap()
    candidates = list(business_users_map.keys())
    candidates = list(combinations(candidates, 2))
    candidates = sc.parallelize(
        list(filter(lambda x: check_n_neighbors(business_users_map[x[0]], business_users_map[x[1]]), candidates)))
    result = candidates.map(lambda x: (x, similarity(business_users_map[x[0]], business_users_map[x[1]]))).filter(
        lambda x: x[1] > 0).map(lambda x: (x[0][0], x[0][1], x[1])).collect()
    with open(sys.argv[2], 'w') as fopen:
        for each in result:
            result = {"b1": each[0], "b2": each[1], "sim": each[2]}
            json.dump(result, fopen)
            fopen.write("\n")
    fopen.close()
else:
    Bands = 40
    rows = 1
    reviews = sc.textFile(input_file).map(lambda row: json.loads(row)).map(
        lambda kv: (kv["user_id"], kv["business_id"], kv["stars"]))
    user_ids = reviews.map(lambda x: x[0]).distinct().sortBy(lambda item: item).zipWithIndex()
    ids_user = user_ids.map(lambda x: (x[1], x[0])).collectAsMap()
    user_ids = user_ids.collectAsMap()
    business_ids = reviews.map(lambda x: x[1]).distinct().sortBy(lambda item: item).zipWithIndex().collectAsMap()
    business_user_map = reviews.map(lambda x: (business_ids[x[1]], (user_ids[x[0]], x[2]))) \
        .groupByKey().map(lambda x: (x[0], list(x[1]))).filter(lambda x: len(x[1]) >= 3).persist()
    m = len(business_ids)
    candidates = business_user_map.flatMapValues(f1).map(lambda x: (x[1][0], hash_ids(x[0]))) \
        .reduceByKey(lambda x, y: list(map(min, zip(x, y))))
    candidates = candidates.map(lambda x: (x[0], f2(x[1]))).flatMapValues(f1).map(lambda x: ((x[1][0], x[1][1]), x[0])). \
        groupByKey().map(lambda x: sorted(set(x[1]))).filter(lambda x: len(x) > 1).flatMap(
        lambda x: combinations(x, 2)).distinct()
    user_business_map = business_user_map.flatMapValues(f1).map(lambda x: (x[1][0], (x[0], x[1][1]))).groupByKey().map(
        lambda x: (x[0], list(set(x[1])))).filter(lambda x: len(x[1]) >= 3).map(
        lambda x: (x[0], f(x[1]))).collectAsMap()
    result = candidates.filter(
        lambda x: jaccard(x)).map(lambda x: (x, similarity(user_business_map[x[0]], user_business_map[x[1]]))) \
        .filter(lambda x: x[1] > 0).map(lambda x: (ids_user[x[0][0]], ids_user[x[0][1]], x[1])).collect()
    print(len(result))
    with open(sys.argv[2], 'w') as fopen:
        for each in result:
            result = {"u1": each[0], "u2": each[1], "sim": each[2]}
            json.dump(result, fopen)
            fopen.write("\n")
    fopen.close()
print("Duration:", time.time() - start)
