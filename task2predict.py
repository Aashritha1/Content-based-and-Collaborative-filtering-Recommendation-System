import json
import math
import sys
import time
from pyspark import SparkContext


def cosine_similarity(user, business):
    users = set(user_profiles.get(user, ''))
    businesses = set(business_profiles.get(business, ''))
    if len(users) and len(businesses):
        return len(users & businesses) / (math.sqrt(len(users)) * math.sqrt(len(businesses)))
    else:
        return 0.0


start = time.time()
input_file = sys.argv[1]
model_file = sys.argv[2]
sc = SparkContext("local[*]")
reviews = sc.textFile(input_file).persist().map(lambda x: json.loads(x)).persist()
with open(model_file) as json_file:
    model = json.load(json_file)
    user_profiles = model["user_profiles"]
    business_profiles = model["business_profiles"]
user_ids = list(user_profiles.keys())
business_ids = list(business_profiles.keys())
predict = reviews.map(lambda x: (x["user_id"], x["business_id"])).filter(
    lambda x: (x[0] in user_ids and x[1] in business_ids)).map(lambda x: (x, cosine_similarity(x[0], x[1]))).filter(
    lambda x: x[1] >= 0.01).map(lambda x: (x[0][0], x[0][1], x[1])).collect()
with open(sys.argv[3], 'w') as fopen:
    for i in range(len(predict)):
        result = {"user_id": predict[i][0], "business_id": predict[i][1], "sim": predict[i][2]}
        json.dump(result, fopen)
        fopen.write("\n")
fopen.close()
print("Duration:", time.time() - start)
