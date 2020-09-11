import json
import sys
import time
from pyspark import SparkContext


def predict(x, model, averages):
    res = []
    for each, every in x[1]:
        value = model.get(tuple((x[0], each)), 0)
        if value == 0:
            value = model.get(tuple((each, x[0])), 0)
        if cf_type == "user_based":
            every = every - averages.get(each, 3.823989)
        res.append(tuple((every, value)))
    if cf_type == "item_based":
        res = sorted(res, key=lambda x: x[1], reverse=True)[0:3]
    num = sum(map(lambda x: x[0] * x[1], res))
    den = sum(map(lambda x: abs(x[1]), res))
    if num == 0 or den == 0:
        return averages.get(x[0], 3.823989)
    if cf_type == "item_based":
        return num/den
    else:
        return averages.get(x[0], 3.823989) + (num / den)


start = time.time()
input_train_file = sys.argv[1]
input_test_file = sys.argv[2]
model_file = sys.argv[3]
cf_type = sys.argv[5]
bus_avg_file = "../resource/asnlib/publicdata/business_avg.json"
user_avg_file = "../resource/asnlib/publicdata/user_avg.json"
sc = SparkContext("local[*]")
reviews = sc.textFile(input_train_file).persist().map(lambda x: json.loads(x)).map(
    lambda x: (x["user_id"], x["business_id"], x["stars"])).persist()
user_ids = reviews.map(lambda x: x[0]).distinct().collect()
business_ids = reviews.map(lambda x: x[1]).distinct().collect()

if cf_type == "item_based":
    model = sc.textFile(model_file).map(lambda x: json.loads(x)).map(lambda x: {(x["b1"], x["b2"]): x["sim"]}). \
        flatMap(lambda x: x.items()).collectAsMap()
    user_bus_data = reviews.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().map(lambda x: (x[0], list(set(x[1]))))
    test_reviews = sc.textFile(input_test_file).persist().map(lambda x: json.loads(x)). \
        map(lambda x: (x["user_id"], x["business_id"])).filter(lambda x: x[0] in user_ids and x[1] in business_ids)
    business_avg = sc.textFile(bus_avg_file).map(lambda x: json.loads(x)).map(lambda x: dict(x)). \
        flatMap(lambda x: x.items()).collectAsMap()
    result = test_reviews.join(user_bus_data).map(lambda x: (x[0], x[1][0], predict(x[1], model, business_avg))).collect()
    with open(sys.argv[4], 'w') as fopen:
        for each in result:
            result = {"user_id": each[0], "business_id": each[1], "stars": each[2]}
            json.dump(result, fopen)
            fopen.write("\n")
    fopen.close()
else:
    model = sc.textFile(model_file).map(lambda x: json.loads(x)).map(
        lambda x: {(x["u1"], x["u2"]): x["sim"]}).flatMap(lambda x: x.items()).collectAsMap()
    bus_user_data = reviews.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().map(lambda x: (x[0], list(set(x[1]))))
    test_reviews = sc.textFile(input_test_file).persist().map(lambda x: json.loads(x)). \
        map(lambda x: (x["business_id"], x["user_id"])).filter(lambda x: x[0] in business_ids and x[1] in user_ids)
    user_avg = sc.textFile(user_avg_file).map(lambda x: json.loads(x)).map(lambda x: dict(x)).\
        flatMap(lambda x: x.items()).collectAsMap()
    result = test_reviews.join(bus_user_data).map(lambda x: (x[0], x[1][0], predict(x[1], model, user_avg))).collect()
    with open(sys.argv[4], 'w') as fopen:
        for each in result:
            result = {"user_id": each[0], "business_id": each[1], "stars": each[2]}
            json.dump(result, fopen)
            fopen.write("\n")
    fopen.close()
print("Duration:", time.time() - start)
