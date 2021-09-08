import json 
import random

# with open('/data3/private/clsi/AmbigQA/data/squad1.1/train.json', 'r') as f:
#     test = json.load(f)['data']

# questions = []
# for d in test:
#     para = d["paragraphs"]
#     for d_ in para:
#         qas = d_["qas"]
#         # print (qas)
#         for d__ in qas:
#             questions.append({"id": d__["id"], "question": d__["question"], "answer": [d__['answers'][0]['text']]})

# random.shuffle(questions)
# dev_len = len(questions) // 10
# print (dev_len)

# with open('/data3/private/clsi/AmbigQA/data/squad_open/dev.json', 'w') as f:
#     json.dump(questions[ : dev_len], f)


# with open('/data3/private/clsi/AmbigQA/data/squad_open/train.json', 'w') as f:
#     json.dump(questions[dev_len : ], f)


with open('/data3/private/clsi/AmbigQA/data/squad_open/train.json', 'r') as f:
    data = json.load(f)

print (len(data))
