import json
import sys
import random
import time
from itertools import combinations
from pyspark import SparkContext


p = 9999999967
a = random.sample(range(1, p), 60)
b = random.sample(range(0, p), 60)


def hash_ids(x):
    hashed = []
    for i in range(60):
        hashed.append((((a[i] * x) + b[i]) % p) % m)
    return hashed


def f(x, maps):
    business = []
    for i in x:
        business.append(maps[i])
    return business


def f1(x):
    return x


def f2(x):
    partition = []
    for i, each in enumerate(range(0, Bands, rows)):
        partition.append((i, tuple(x[each:each + rows])))
    return partition


def jaccard(x):
    sim_value = 0.0
    try:
        column1 = business_user_id[x[0]]
        column2 = business_user_id[x[1]]
        sim_value = len(column1 & column2) / len(column1 | column2)
    except:
        sim_value = 0.0
    return map_ids_business[x[0]], map_ids_business[x[1]], sim_value


start = time.time()
input_file = sys.argv[1]
sc = SparkContext("local[*]")
reviews = sc.textFile(input_file).persist().map(lambda x: json.loads(x)).map(
    lambda x: (x["user_id"], x["business_id"]))
user_business_id = reviews.groupByKey().map(lambda x: (x[0], set(x[1])))
business_user_id = reviews.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], set(x[1])))
user_ids = user_business_id.map(lambda x: x[0]).sortBy(lambda x: x).zipWithIndex()
users_maps = user_ids.collectAsMap()
business_ids = business_user_id.map(lambda x: x[0]).sortBy(lambda x: x).zipWithIndex()
map_ids_business = business_ids.map(lambda x: (x[1], x[0])).collectAsMap()
business_maps = business_ids.collectAsMap()
m = len(users_maps)
user_hashes = user_ids.map(lambda x: (users_maps[x[0]], hash_ids(x[1])))
user_business_id = user_business_id.map(lambda x: (users_maps[x[0]], f(x[1], business_maps)))
business_user_id = business_user_id.map(lambda x: (business_maps[x[0]], set(f(x[1], users_maps)))).collectAsMap()
users_signatures = user_business_id.join(user_hashes).map(lambda x: (x[1][1], x[1][0])).flatMapValues(f1).map(
    lambda x: (x[1], x[0])).reduceByKey(lambda x, y: list(map(min, zip(x, y))))
Bands = 60
rows = 1
user_candidates = users_signatures.map(lambda x: (x[0], f2(x[1]))).flatMapValues(f1).map(
    lambda x: ((x[1][0], x[1][1][0]), x[0])).groupByKey().map(lambda x: list(x[1])).filter(
    lambda x: len(x) > 1).flatMap(lambda x: combinations(x, 2)).distinct()
similarity_check = (user_candidates.map(lambda x: jaccard(x))).filter(lambda x: x[2] >= 0.05).collect()
with open(sys.argv[2], 'w') as fopen:
    for i in range(len(similarity_check)):
        result = {"b1": similarity_check[i][0], "b2": similarity_check[i][1], "sim": similarity_check[i][2]}
        json.dump(result, fopen)
        fopen.write("\n")
fopen.close()
print("Duration:", time.time()-start)
