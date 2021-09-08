# Answer Equivalence for Open=Domain Question Answering

This repo contains the code to reproduce the results in the our EMNLP paper <b>What’s in a Name? Answer Equivalence For Open-Domain Question Answering</b>. The codebase is largely adapted from [Min et al.](https://github.com/shmsw25/AmbigQA/tree/master/codes). All our experiments are done with BERT-base-uncased as the backbone. 


## Content
1. [Dependencies](#Dependencies)
2. [Download data](#download-data)
3. [QA Pipeline](#qa-pipeline)

## Dependencies

The dependecies are essentially the same as the original repo of [Min et al.](https://github.com/shmsw25/AmbigQA/tree/master/codes). Specially, the experiments in our paper are run with `transformers==3.4.0` and `tokenizers==0.9.2`. I am not sure whether other versions are compatible or would produce the same results. 

## Download data
We used three datasets in our paper: NQ, TriviaQA and SQuAD. Note that we treat SQuAD as an open-domain QA setting and retrieve passages from Wikipedia for each question instead of using the provided ones. 

In our answer augmentation, we append all alias entities of the original answer entity to the correct answer list. We then unify the format of all these three datasets so that the training can be run easily with the same data processor and training command. 

To save you the trouble of these preprocessing, we directly provide all the <b>processed data</b>. After you download and unzip the file, you should see these directories:

-`nqopen`: the original NQ dataset. 

-`nq_multi`: the augmented NQ dataset.

-`triviaqa`: the original TriviaQA dataset. Note that in this version we only take the original <b>single</b> gold answer (the string in the `data["Answer"]["Value"]` field in the original [TriviaQA release](https://github.com/mandarjoshi90/triviaqa/blob/master/samples/triviaqa_sample.json)). And it does not include the `"Aliases"` field in the answer list. 

-`triviaqa_alias`: the TriviaQA dataset with the answer strings in the `"Aliases"` field of the original TriviaQA released combined into the answer list. These answer aliases are mined from Wikipedia, by the original authors [Mandar et al.](https://github.com/mandarjoshi90/triviaqa).

-`triviaqa_multi`: the TriviaQA dataset augmented with answer aliases from Freebase, this is mined by ourselves, as a comparison to the original answer aliases in Mandar et al.

-`squad_open`: the original SQuAD dataset, without paired passages.

-`squad_open_multi`: the SQuAD dataset augmented with answer aliases from Freebase.

In each directory, you should see the train/dev/test.json files. In the answer list for each question, the first answer string is always the original gold answer, and all answers following that are its aliases mined from knowledge bases. You can take a look at the actual data and see how the answer aliases look like, and be aware that some aliases may be noisy (it's just impossible to have all aliases being correct aliases to the answers), we have provided a detailed analysis in the paper showing that in most cases the noises are acceptable and doesn't impact the actual performance much. 

We have also provided the tokenized files of those data, so that you can skip the tokenization part. But if you are not using `bert-base-uncased`, then you need to tokenize them by yourselves with your own tokenizer.

In case you are wondering what if we combine the aliases from both Wikipedia and Freebase for TriviaQA, we have tried that, and found that it gives some additional gains during augmented evaluation, but no gains in augmented training as compared to just using one source of aliases alone. 


## QA Pipeline

We follow the same open-domain QA pipeline as in [Min et al.](https://github.com/shmsw25/AmbigQA/tree/master/codes). Note that you need to do the passage retriveal for every dataset instead of reusing the retrieved passage indices of the original dataset for the augmented dataset, because their gold answers are different. 

After that, you can start training. Here is an example of the training script that we used:
```
python3 cli.py \
--seed 12 \
--do_train \
--do_predict \
--task qa \
--num_train_epochs 4 \
--dpr_data_dir $dpr_data_dir \
--output_dir out/reader_nq_multi_new \
--train_file $nq_train \
--predict_file $nq_multi_dev \
--bert_name bert-base-uncased \
--train_M 24 \
--test_M 10 \
--train_batch_size 4 \
--gradient_accumulation_steps 2 \
--predict_batch_size 32 \
--eval_period 800 \
--wait_step 10 \
--topk_answer 10 \
--learning_rate 3e-5 \
--checkpoint /home/sichenglei/bert-base-uncased/pytorch_model.bin
```

And to test on the test set:
```
python3 cli.py \
--seed 12 \
--do_predict \
--task qa \
--dpr_data_dir $dpr_data_dir \
--output_dir out/reader_nq_multi_new \
--train_file $nq_train \
--predict_file $nq_multi_test \
--bert_name bert-base-uncased \
--train_M 24 \
--test_M 10 \
--predict_batch_size 32 \
--eval_period 800 \
--wait_step 10 \
--topk_answer 10 \
--checkpoint out/reader_nq_multi_new/best-model.pt
```

We used these same hyper-parameters for all datasets. 
