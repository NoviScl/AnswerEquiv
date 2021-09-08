import json
import numpy as np

# with open("/data3/private/clsi/AmbigQA/data/nqopen/train_for_infer/dev-uncased-BertTokenized.json", "r") as f:
#     data = json.load(f)
# print (len(data[0]))

with open("/data3/private/clsi/AmbigQA/data/nqopen/train.json", "r") as f:
    nq = json.load(f)

print (len(nq))

with open("preds_nq_infer.log", "r") as f:
    preds = f.readlines()

preds = [line.strip() for line in preds if "[{'text':" in line]

print (len(preds))

thres = np.log(0.08)

new_ambig_nq = []

for i in range(len(nq)):
    d_nq = nq[i]
    ans = d_nq["answer"][:]
    id = d_nq["id"]
    qn = d_nq["question"]

    d_ambig = {}
    preds_i = eval(preds[i])
    for d in preds_i:
        if d['log_softmax'] > thres and d['text'] not in ans:
            ans.append(d['text'])
    
    if i%2000 == 0:
        print (len(d_nq["answer"]), len(ans))

    d_ambig["annotations"] = [{"type": "singleAnswer", "answer": ans}]
    d_ambig["id"] = id
    d_ambig["question"] = qn

    new_ambig_nq.append(d_ambig)

print (len(new_ambig_nq))

with open("/data3/private/clsi/AmbigQA/data/ambig_nq/train_light.json", "w") as f:
    json.dump(new_ambig_nq, f)

# with open("out/reader_nq_infer/dev_predictions_BertTokenized.json", "r") as f:
#     data = json.load(f)
# print (len(data['positive_input_ids']))

# print (len(data))

# with open("preds_nq_probe.log", "r") as f:
#     data = f.readlines()

# counter = 0
# dic = {}
# thres = np.log(0.08)
# for line in data:
#     if "[{'text':" in line:
#         lst = eval(line.strip())
#         counter += 1
#         c = 0
#         for d in lst:
#             if d['log_softmax'] > thres:
#                 c += 1
#         if c not in dic:
#             dic[c] = 1
#         else:
#             dic[c] += 1

# print (counter)
# print (dic)

# lst = []

# for line in data:
#     if "[{'text':" in line:
#         d = eval(line.strip())
