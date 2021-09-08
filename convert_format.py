import json 

dev_data = json.load(open("/data3/private/clsi/AmbigQA/data/ambigqa/dev_light.json", "r"))
dpr_preds = json.load(open("out/reader_ambig/dev_eval_predictions.json", "r"))
# print (dev_data[0]["question"])

new_list = {}
for correct, pred in zip(dev_data, dpr_preds):
    answers = []
    for annotation in correct["annotations"]:
        if annotation["type"]=="singleAnswer":
            answers.append([list(set(annotation["answer"]))])
        else:
            answers.append([list(set(pair["answer"])) for pair in annotation["qaPairs"]])
    answers = [a for anno in answers for pair in anno for a in pair]
    answers = list(set(answers))

    # new_json = {}
    # new_json["qid"] = correct["id"]
    # new_json["question"] = correct["question"]
    # new_json["predicted"] = [pred[0][0]["text"]]
    # new_json["correct_answer"] = answers
    # new_json["context"] = pred[0][0]["passage"]
    # print (pred[0][0])
    # break
    # new_list.append(new_json)
    new_list[correct["id"]] = [pred[0][0]["text"]]

json.dump(new_list, open("out/reader_ambig/eval_top1_dev.json", "w"), indent=4)