import os
import pickle as pkl
import gzip
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from ambigqa_evaluate_script import normalize_answer
from DataLoader import MyDataLoader

class PassageData(object):
    def __init__(self, logger, args, tokenizer):
        self.logger = logger
        self.args = args
        self.data_path = os.path.join(args.dpr_data_dir,
                                      "data/wikipedia_split/psgs_w100{}.tsv.gz".format("_20200201" if args.wiki_2020 else ""))

        self.passages = None
        self.titles = None
        self.tokenizer = tokenizer
        self.tokenized_data = None

    def load_db(self):
        if not self.args.skip_db_load:
            data = []
            with gzip.open(self.data_path, "rb") as f:
                for line in tqdm(f):
                    data.append(line.decode().strip().split("\t"))
                    if self.args.debug and len(data)==100:
                        break
            assert all([len(d)==3 for d in data])
            assert data[0]==["id", "text", "title"]
            self.passages = {int(d[0])-1:d[1].lower() for d in data[1:]}
            self.titles = {int(d[0])-1:d[2].lower() for d in data[1:]}
            self.logger.info("Loaded {} passages".format(len(self.passages)))

    def load_tokenized_data(self, model_name, all=False, do_return=False, index=None):
        if all:
            tokenized_data = {"input_ids": [], "attention_mask": []}
            all_idx = 10
            if self.args.wiki_2020:
                all_idx = 12
            for index in range(all_idx):
                curr_tokenized_data = self.load_tokenized_data(model_name, all=False, do_return=True, index=index)
                tokenized_data["input_ids"] += curr_tokenized_data["input_ids"]
                tokenized_data["attention_mask"] += curr_tokenized_data["attention_mask"]
        else:
            index=self.args.db_index if index is None else index
            assert 0<=index<12
            if model_name=="bert":
                cache_path = self.data_path.replace(".tsv.gz", "_{}_BertTokenized.pkl".format(index))
            elif model_name=="albert":
                cache_path = self.data_path.replace(".tsv.gz", "_{}_AlbertTokenized.pkl".format(index))
            elif model_name=="bart":
                cache_path = self.data_path.replace(".tsv.gz", "_{}_BartTokenized.pkl".format(index))
            else:
                raise NotImplementedError(model_name)
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.logger.info("Loading tokenized passage data from cache: {}".format(cache_path))
                    tokenized_data = pkl.load(f)
            else:
                assert not self.args.skip_db_load
                if self.titles is None or self.passages is None:
                    self.load_db()
                # tokenize 2.2M for each thread
                min_idx = index*2200000
                max_idx = min(len(self.titles), (index+1)*2200000)
                self.logger.info("Start tokenizing from {} to {}".format(min_idx, max_idx))
                input_data = [self.titles[_id] + " " + self.tokenizer.sep_token + " " + self.passages[_id]
                            for _id in range(min_idx, max_idx)]
                tokenized_data = self.tokenizer.batch_encode_plus(input_data,
                        max_length=128,
                        pad_to_max_length=model_name in ["albert", "bert"])
                with open(cache_path, "wb") as f:
                    pkl.dump({"input_ids": tokenized_data["input_ids"],
                              "attention_mask": tokenized_data["attention_mask"]}, f)

        self.tokenized_data = tokenized_data
        self.logger.info("Finish loading {} {} passage tokenized data".format(len(tokenized_data["input_ids"]), model_name))
        if do_return:
            return tokenized_data

    def load_dataset(self, model_name, do_return=False):
        if self.tokenized_data is None:
            self.load_tokenized_data("bert", index=self.args.db_index)
        tokenized_data = self.tokenized_data
        assert tokenized_data is not None
        input_ids = torch.LongTensor(tokenized_data["input_ids"])
        attention_mask = torch.LongTensor(tokenized_data["attention_mask"])
        print (model_name, input_ids.size(), attention_mask.size())
        self.dataset = TensorDataset(input_ids, attention_mask)
        if do_return:
            return self.dataset

    def load_dataloader(self, batch_size, is_training=None, do_return=False):
        self.dataloader = MyDataLoader(self.args,
                                       self.dataset,
                                       batch_size=int(batch_size/4),
                                       is_training=self.is_training if is_training is None else is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions, answers):
        if self.args.skip_db_load:
            return [0]
        if self.passages is None:
            self.load_db()
        assert len(predictions)==len(answers)
        assert not self.args.skip_db_load
        recall = defaultdict(list)
        k_list = [10, 20, 50, 100, 1000]
        for (pids, answer) in zip(predictions, answers):
            passages = [normalize_answer(self.passages[pid]) for pid in pids]
            answer = list(set([normalize_answer(a) for annotation in answer for answer_list in annotation for a in answer_list])) # flatten each answer list to one layer list and normalize 
            curr_recall = [any([ans in p for ans in answer]) for p in passages]
            for k in k_list:
                recall[k].append(any(curr_recall[:k]))
        for k in k_list:
            self.logger.info("Recall @ %d\t%.2f" % (k, np.mean(recall[k])))
        return recall[100]


