import os
import numpy as np
import torch

from tqdm import tqdm
from transformers import BartTokenizer, AlbertTokenizer, BertTokenizer
from transformers import BartConfig, AlbertConfig, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from QAData import QAData, AmbigQAData
from QGData import QGData, AmbigQGData
from PassageData import PassageData
from tqdm import tqdm 

from models.span_predictor import SpanPredictor, AlbertSpanPredictor
from models.seq2seq import MyBart
from models.seq2seq_with_prefix import MyBartWithPrefix
from models.biencoder import MyBiEncoder


import pickle
import unicodedata
from tqdm import tqdm
import json
import re
from urllib.parse import unquote
from fuzzywuzzy import fuzz, process as fuzzy_process
from bisect import bisect_left
import numpy as np 

def normalize(text):
    return unicodedata.normalize('NFD', text).replace(' ', '_').lower()


def find_spans(tokenizer, answers, context_tokens):
    context_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    extracted_spans = []
    already_found_ans = []
    for ans in answers:
        ans_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ans))
        for j in range(len(context_ids) - len(ans_id) + 1):
            if context_ids[j : j+len(ans_id)] == ans_id:
                if ((j, j+len(ans_id)-1) not in extracted_spans) and (ans not in already_found_ans):
                    extracted_spans.append((j, j+len(ans_id)-1))
                    already_found_ans.append(ans)
    return extracted_spans, already_found_ans


# def find_spans_remove(tokenizer, answers, context_tokens):
#     context_ids = tokenizer.convert_tokens_to_ids(context_tokens)
#     extracted_spans = []
#     for ans in answers:
#         ans_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ans))
#         for j in range(len(context_ids) - len(ans_id) + 1):
#             if context_ids[j : j+len(ans_id)] == ans_id:
#                 if (j, j+len(ans_id)-1) not in extracted_spans:
#                     extracted_spans.append((j, j+len(ans_id)-1))
#                     return extracted_spans, ans
#     return extracted_spans, None


## to be used in cli.py like run.py, to build the graphs
def build_graph(args, logger):
    args.dpr = args.task=="dpr"
    args.is_seq2seq = 'bart' in args.bert_name
    if 'bart' in args.bert_name:
        tokenizer = BartTokenizer.from_pretrained(args.bert_name)
        tokenizer.add_tokens(["<SEP>"])
        Model = MyBartWithPrefix if args.do_predict and args.nq_answer_as_prefix else MyBart
        Config = BartConfig
        args.append_another_bos = True
    elif 'albert' in args.bert_name:
        tokenizer = AlbertTokenizer.from_pretrained(args.bert_name)
        Model = AlbertSpanPredictor
        Config = AlbertConfig
    elif 'bert' in args.bert_name:
        tokenizer = BertTokenizer.from_pretrained(args.bert_name)
        Model = MyBiEncoder if args.dpr else SpanPredictor
        Config = BertConfig
    else:
        raise NotImplementedError()
    
    if args.dpr:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        Model = MyBiEncoder
        args.checkpoint = os.path.join(args.dpr_data_dir, "checkpoint/retriever/multiset/bert-base-encoder.cp")
        assert not args.do_train, "Training DPR is not supported yet"

    passages = PassageData(logger, args, tokenizer)
    passages.load_db()
    wiki_pass = passages.passages
    
    ## load AmbigQA data
    with open(args.predict_file, 'r') as f:
        dataset = json.load(f)

    # dpr_predictions = "out/dpr/train_for_inference_predictions.json"
    # dpr_predictions = "out/dpr/dev_predictions.json"
    dpr_predictions = "/home/sichenglei/OpenQA/reranking_results_AmbigQA/ambigqa_train_2020.json"
    with open(dpr_predictions, 'r') as f:
        dpr_preds = json.load(f)
    # with open("/home/sichenglei/OpenQA/entity_rank/rerank_retrieve_train_top10.pkl", "rb") as f:
    #     dpr_preds_list = pickle.load(f)
    # print (len(dataset), len(dpr_preds_list))

    # dpr_preds = {}
    # for d in dpr_preds_list:
    #     dpr_preds[d["qid"]] = d["passages"]
    
    # anno_d = []
    # for anno in dataset:
    #     anno_d.append(anno["id"])
    # anno_d = list(set(anno_d))
    # print (len(anno_d))

    print (len(dataset), len(dpr_preds))
    # print (len(dpr_preds['1135104210639387334']))
    assert len(dataset) == len(dpr_preds), "Number of samples mismatch."
    train_graph = list()

    no_ans_counter = 0
    ## passages.passages is the dict storing all passages, access by index
    for i in tqdm(range(len(dataset))):
        annotation = dataset[i] 
        qid = annotation["id"]
        answer = []

        ## for ambigqa
        ## get the set of correct answers
        for anno in annotation["annotations"]:
            if anno["type"] == "singleAnswer":
                answer.extend(anno["answer"])
            else:
                for qa in anno["qaPairs"]:
                    answer.extend(qa["answer"])

        ## for NQ
        # answer = annotation["answer"]
        
        answer = list(set(answer))

        graph = {"qid": qid, "question": annotation["question"], "node": list(), "edge": list()}
        node_num = 0

        ## potential problem: there can be multiple answers in the same passage
        passages = dpr_preds[i][:10] # take top 10 retrieved passages for now
        # passages = dpr_preds[qid]
        # print (len(dpr_preds[i]))
        has_ans = 0 
        for p in passages:
            wiki_p = wiki_pass[p]
            # wiki_p = p
            is_ans = 0
            # for ans in answer:
                ## clsi: this step is actually unnecessary, if extracted_span is None then no ans
                # if normalize(ans) in normalize(wiki_p):
                #     is_ans = 1
                # else:
                #     is_ans = 0
            context_tokens = tokenizer.tokenize(wiki_p)
            # if is_ans == 1:
            #     # extracted_spans = find_start_end_after_tokenized(tokenizer, context_tokens, spans = answer)
            #     extracted_spans = find_spans(tokenizer, answer, context_tokens)
            #     if extracted_spans == []:
            #         # print ("Failed to find the answer spans!!")
            #         is_ans = 0
            #     # print (context_tokens, answer, extracted_spans)
            # else:
            #     extracted_spans = []
            extracted_spans, ans_ = find_spans(tokenizer, answer, context_tokens)
            if extracted_spans == []:
                is_ans = 0
            else:
                is_ans = 1
            has_ans += is_ans
            graph["node"].append({"node_id": node_num, "context": context_tokens, "spans": extracted_spans, "is_ans": is_ans})
            node_num += 1
        
        if has_ans == 0:
            no_ans_counter +=1
            # continue ## filter no ans cases

        for node_id in range(node_num):
            for node_id_2 in range(node_num):
                graph["edge"].append({"start": node_id, "end": node_id_2})
                # if node_id != node_id_2:
                #     graph["edge"].append({"start": node_id, "end": node_id_2})
        
        train_graph.append(graph)
 
                     
    # print(len(train_graph))
    # print (no_ans_counter)
    print ("Recall: {}/{}={}%".format(len(train_graph) - no_ans_counter, len(train_graph), (len(train_graph) - no_ans_counter) / len(train_graph)))
    # pickle.dump(train_graph, open('/home/sichenglei/OpenQA/ambig_dev_graph_p10_allConnect.pkl', 'wb'))
    # pickle.dump(train_graph, open('/home/sichenglei/OpenQA/nq_graph/test_graph.pkl', 'wb'))
    pickle.dump(train_graph, open('/home/sichenglei/OpenQA/ambig_reranked_graph/train_graph.pkl', 'wb'))
            


# build training and dev corpus for re-ranker training and eval
def build_rerank(args, logger):
    args.dpr = args.task=="dpr"
    args.is_seq2seq = 'bart' in args.bert_name
    if 'bart' in args.bert_name:
        tokenizer = BartTokenizer.from_pretrained(args.bert_name)
        tokenizer.add_tokens(["<SEP>"])
        Model = MyBartWithPrefix if args.do_predict and args.nq_answer_as_prefix else MyBart
        Config = BartConfig
        args.append_another_bos = True
    elif 'albert' in args.bert_name:
        tokenizer = AlbertTokenizer.from_pretrained(args.bert_name)
        Model = AlbertSpanPredictor
        Config = AlbertConfig
    elif 'bert' in args.bert_name:
        tokenizer = BertTokenizer.from_pretrained(args.bert_name)
        Model = MyBiEncoder if args.dpr else SpanPredictor
        Config = BertConfig
    else:
        raise NotImplementedError()
    
    if args.dpr:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        Model = MyBiEncoder
        args.checkpoint = os.path.join(args.dpr_data_dir, "checkpoint/retriever/multiset/bert-base-encoder.cp")
        assert not args.do_train, "Training DPR is not supported yet"

    passages = PassageData(logger, args, tokenizer)
    passages.load_db()
    wiki_pass = passages.passages

    train_file = "/data3/private/clsi/AmbigQA/data/ambigqa/train_light.json"
    dev_file = "/data3/private/clsi/AmbigQA/data/ambigqa/dev_light.json"
    
    # ## load AmbigQA data
    # with open(args.predict_file, 'r') as f:
    #     dataset = json.load(f)
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    with open(dev_file, 'r') as f:
        dev_data = json.load(f)
    

    # dpr_predictions = "out/dpr/train_for_inference_predictions.json"
    # dpr_predictions = "out/dpr/dev_predictions.json"
    # with open(dpr_predictions, 'r') as f:
    #     dpr_preds = json.load(f)
    # train_preds = "out/dpr/train_for_inference_predictions.json"
    train_preds = "out/dpr_1k/train_for_inference_predictions.json"
    dev_preds = "out/dpr_1k/dev_predictions.json"
    with open(train_preds, "r") as f:
        dpr_train_preds = json.load(f)
    with open(dev_preds, "r") as f:
        dpr_dev_preds = json.load(f)
    
    
    assert len(train_data) == len(dpr_train_preds), "Number of train samples mismatch."
    assert len(dev_data) == len(dpr_dev_preds), "Number of dev samples mismatch."
    # train_graph = list()
    train_list = []
    dev_list = []

    # stats: for train set only
    ans_counter = 0
    total = 0
    ## passages.passages is the dict storing all passages, access by index
    ## build train set
    for i in tqdm(range(len(train_data))):
        annotation = train_data[i] 
        qid = annotation["id"]
        answer = []

        ## get the set of correct answers
        for anno in annotation["annotations"]:
            if anno["type"] == "singleAnswer":
                answer.extend(anno["answer"])
            else:
                for qa in anno["qaPairs"]:
                    answer.extend(qa["answer"])
        
        answer_ = list(set(answer))
        ## deduplicate answers
        answer = []
        for i in range(len(answer_)):
            unique = True
            for j in range(len(answer_)):
                if i!=j and answer_[i].lower() in answer_[j].lower():
                    unique = False
                    break
            if unique:
                answer.append(answer_[i])
                

        passages = dpr_train_preds[i][:100] # take top 100 retrieved passages for train
        assert len(passages) == 100, "Number of retrieved passages mismatch"
        for p in passages:
            wiki_p = wiki_pass[p]
            p_dict = {}

            context_tokens = tokenizer.tokenize(wiki_p)
            extracted_spans, ans_ = find_spans(tokenizer, answer, context_tokens)
            if extracted_spans == []:
                is_ans = 0
            else:
                is_ans = 1

            p_dict["qid"] = qid 
            p_dict["question"] = annotation["question"]
            p_dict["passage"] = wiki_p 
            p_dict["label"] = is_ans
            
            total += 1
            ans_counter += is_ans 

            train_list.append(p_dict)
        
    print ("Train: positive ratio: {}/{}={}%".format(ans_counter, total, ans_counter/total*100))
    pickle.dump(train_list, open('/home/sichenglei/OpenQA/entity_rank/rerank_train_200.pkl', 'wb'))


## to construct the data for training RNN reranker
def build_path_data(args, logger):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    passages = PassageData(logger, args, tokenizer)
    passages.load_db()
    wiki_pass = passages.passages

    train_file = "/data3/private/clsi/AmbigQA/data/ambigqa/train_light.json"
    dev_file = "/data3/private/clsi/AmbigQA/data/ambigqa/dev_light.json"

    with open(train_file, 'r') as f:
        train_data = json.load(f)
    with open(dev_file, 'r') as f:
        dev_data = json.load(f)
    
    # note: for AmbigQA, we load the 20200201 retrieved results!
    train_preds = "out/reader_ambig/train_for_inference_20200201_predictions.json"
    dev_preds = "out/reader_ambig/dev_20200201_predictions.json"
    with open(train_preds, "r") as f:
        dpr_train_preds = json.load(f)
    with open(dev_preds, "r") as f:
        dpr_dev_preds = json.load(f)    
    
    assert len(train_data) == len(dpr_train_preds), "Number of train samples mismatch."
    assert len(dev_data) == len(dpr_dev_preds), "Number of dev samples mismatch."
    train_list = []
    dev_list = []

    # stats: for train set only
    pos_counter = 0
    total = 0
    multi_ans = 0
    ## passages.passages is the dict storing all passages, access by index
    ## build train set
    for i in tqdm(range(len(dev_data))):
        annotation = dev_data[i] 
        qid = annotation["id"]
        q_dict = {}
        q_dict["q_id"] = qid
        q_dict["question"] = annotation["question"]
        answer = []

        ## get the set of correct answers
        for anno in annotation["annotations"]:
            if anno["type"] == "singleAnswer":
                answer.extend(anno["answer"])
            else:
                for qa in anno["qaPairs"]:
                    answer.extend(qa["answer"])
        
        answer = list(set(answer))
        # answer_ = list(set(answer))
        # ## deduplicate answers
        # answer = []
        # for i in range(len(answer_)):
        #     unique = True
        #     for j in range(len(answer_)):
        #         if i!=j and answer_[i].lower() in answer_[j].lower():
        #             unique = False
        #             break
        #     if unique:
        #         answer.append(answer_[i])
                

        q_dict["answers"] = answer
        q_dict["context"] = {} # a dict: {idx: passage}
        q_dict["short_gold"] = [] # list of positive passages' index, no repeated passages for same answer
        q_dict["long_gold"] = [] # list of all passages containing gold spans, allowing repeating of same answer
        ## for one-step version: short_gold is top1, long_gold is all gold.
        q_dict["negatives"] = [] # save top20 unmatched negatives
        q_dict["ans_dict"] = {}

        passages = dpr_dev_preds[i][:100] # take top 10 retrieved passages for train
        assert len(passages) == 100, "Number of retrieved passages mismatch"
        for pi, p in enumerate(passages):
            wiki_p = wiki_pass[p]
            q_dict["context"][pi] = wiki_p
            # p_dict = {}

            context_tokens = tokenizer.tokenize(wiki_p)
            extracted_spans, ans_ = find_spans(tokenizer, answer, context_tokens)
            if len(ans_) > 1:
                multi_ans += 1
                # print ("Passage: ", wiki_p)
                # print ("Matched gold answers: ", ans_)
                
            # extracted_spans_remove, ans = find_spans_remove(tokenizer, answer_remove, context_tokens)
            if len(extracted_spans) > 0:
                q_dict["long_gold"].append(pi)
                pos_counter += 1
                ## for one_step
                if len(q_dict["short_gold"]) == 0:
                    q_dict["short_gold"].append(pi) # use top-1 match as short_gold
            else:
                # if len(q_dict["negatives"]) < 20:
                q_dict["negatives"].append(pi)

            # if extracted_spans_remove == []:
            #     is_ans = 0
            # else:
            #     try:
            #         answer_remove.remove(ans) ## avoid repeated passages for same ans
            #     except:
            #         print (answer, answer_remove, ans)
            #     is_ans = 1
            #     q_dict["short_gold"].append(pi)
            #     pos_counter += 1

            
            total += 1 
        
        ## construct ans_dict (answer: list of matched passages)
        for ans in answer:
            q_dict["ans_dict"][ans] = []
            for pi, p in enumerate(passages):
                wiki_p = wiki_pass[p]
                context_tokens = tokenizer.tokenize(wiki_p)
                extracted_spans, ans_ = find_spans(tokenizer, [ans], context_tokens)

                if len(extracted_spans) > 0:
                    q_dict["ans_dict"][ans].append(pi)


        train_list.append(q_dict)
    
    # print ("positive ratio: {}/{}={}%".format(pos_counter, total, pos_counter/total*100))
    # print ("multi_ans: ", multi_ans)
    # # pickle.dump(train_list, open('/home/sichenglei/OpenQA/entity_rank/rerank_train_200.pkl', 'wb'))
    # with open('/home/sichenglei/reason_paths/graph_retriever/data_ans_dict/ambig_dev.json', 'w') as f:
    #     json.dump(train_list, f, indent=4)


    # # same for dev, yes I know I should abstract it as a function
    # pos_counter = 0
    # total = 0
    # for i in tqdm(range(len(dev_data))):
    #     annotation = dev_data[i] 
    #     qid = annotation["id"]
    #     q_dict = {}
    #     q_dict["q_id"] = qid
    #     q_dict["question"] = annotation["question"]
    #     answer = []

    #     ## get the set of correct answers
    #     for anno in annotation["annotations"]:
    #         if anno["type"] == "singleAnswer":
    #             answer.extend(anno["answer"])
    #         else:
    #             for qa in anno["qaPairs"]:
    #                 answer.extend(qa["answer"])
        
    #     answer = list(set(answer))
    #     answer_remove = answer[:]

    #     q_dict["answers"] = answer
    #     q_dict["context"] = {} # a dict: {idx: passage}
    #     q_dict["short_gold"] = [] # list of positive passages' index, no repeated passages for same answer
    #     q_dict["long_gold"] = [] # list of all passages containing gold spans, allowing repeating of same answer

    #     passages = dpr_dev_preds[i][:10] # take top 10 retrieved passages for train
    #     assert len(passages) == 10, "Number of retrieved passages mismatch"
    #     for pi, p in enumerate(passages):
    #         wiki_p = wiki_pass[p]
    #         q_dict["context"][pi] = wiki_p
    #         # p_dict = {}

    #         context_tokens = tokenizer.tokenize(wiki_p)
    #         extracted_spans = find_spans(tokenizer, answer, context_tokens)
    #         extracted_spans_remove, ans = find_spans_remove(tokenizer, answer_remove, context_tokens)
    #         if len(extracted_spans) > 0:
    #             q_dict["long_gold"].append(pi)

    #         if extracted_spans_remove == []:
    #             is_ans = 0
    #         else:
    #             try:
    #                 answer_remove.remove(ans) ## avoid repeated passages for same ans
    #             except:
    #                 print (answer, answer_remove, ans)
    #             is_ans = 1
    #             q_dict["short_gold"].append(pi)
    #             pos_counter += 1
            
    #         total += 1 

    #     dev_list.append(q_dict)
        
    # print ("Dev: positive ratio: {}/{}={}%".format(pos_counter, total, pos_counter/total*100))
    # with open('/home/sichenglei/reason_paths/graph_retriever/data_new/ambig_dev.json', 'w') as f:
    #     json.dump(dev_list, f, indent=4)

    



    # total = 0
    # ans_counter = 0

    # for i in tqdm(range(len(dev_data))):
    #     annotation = dev_data[i] 
    #     qid = annotation["id"]
    #     answer = []

    #     ## get the set of correct answers
    #     for anno in annotation["annotations"]:
    #         if anno["type"] == "singleAnswer":
    #             answer.extend(anno["answer"])
    #         else:
    #             for qa in anno["qaPairs"]:
    #                 answer.extend(qa["answer"])
        
    #     answer = list(set(answer))

    #     passages = dpr_dev_preds[i][:1000] # take top 1000 retrieved passages for dev
    #     for p in passages:
    #         wiki_p = wiki_pass[p]
    #         p_dict = {}

    #         context_tokens = tokenizer.tokenize(wiki_p)
    #         extracted_spans = find_spans(tokenizer, answer, context_tokens)
    #         if extracted_spans == []:
    #             is_ans = 0
    #         else:
    #             is_ans = 1

    #         p_dict["qid"] = qid 
    #         p_dict["question"] = annotation["question"]
    #         p_dict["passage"] = wiki_p 
    #         p_dict["label"] = is_ans
            
    #         total += 1
    #         ans_counter += is_ans 

    #         dev_list.append(p_dict)
        
    # print ("Dev: positive ratio: {}/{}={}%".format(ans_counter, total, ans_counter/total*100))
    # pickle.dump(dev_list, open('/home/sichenglei/OpenQA/entity_rank/rerank_dev.pkl', 'wb'))

            







