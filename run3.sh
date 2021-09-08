export CUDA_VISIBLE_DEVICES=0

ambig_train=/data3/private/clsi/AmbigQA/data/ambigqa/train_light.json
ambig_dev=/data3/private/clsi/AmbigQA/data/ambigqa/dev_light.json
nq_train=/data3/private/clsi/AmbigQA/data/nqopen/train.json
nq_dev=/data3/private/clsi/AmbigQA/data/nqopen/dev.json
nq_train_dev=/data3/private/clsi/AmbigQA/data/nqopen/train_for_infer/dev.json
nq_multi_train=/data3/private/clsi/AmbigQA/data/nq_multi/train.json
nq_multi_dev=/data3/private/clsi/AmbigQA/data/nq_multi/dev.json
nq_multi_test=/data3/private/clsi/AmbigQA/data/nq_multi/test.json
nq_test=/data3/private/clsi/AmbigQA/data/nqopen/test.json
dpr_data_dir=/data3/private/clsi/DPR

trivia_train=/data3/private/clsi/AmbigQA/data/triviaqa/train.json 
trivia_dev=/data3/private/clsi/AmbigQA/data/triviaqa/dev.json 
trivia_test=/data3/private/clsi/AmbigQA/data/triviaqa/test.json 


trivia_multi_train=/data3/private/clsi/AmbigQA/data/triviaqa_multi/train.json 
trivia_multi_dev=/data3/private/clsi/AmbigQA/data/triviaqa_multi/dev.json 
trivia_multi_test=/data3/private/clsi/AmbigQA/data/triviaqa_multi/test.json 


trivia_alias_train=/data3/private/clsi/AmbigQA/data/triviaqa_alias/train.json 
trivia_alias_dev=/data3/private/clsi/AmbigQA/data/triviaqa_alias/dev.json 
trivia_alias_test=/data3/private/clsi/AmbigQA/data/triviaqa_alias/test.json 


trivia_union_train=/data3/private/clsi/AmbigQA/data/triviaqa_union/train.json 
trivia_union_dev=/data3/private/clsi/AmbigQA/data/triviaqa_union/dev.json 
trivia_union_test=/data3/private/clsi/AmbigQA/data/triviaqa_union/test.json 


ambig_nq_train=/data3/private/clsi/AmbigQA/data/ambig_nq/train_light.json
ambig_nq_dev=/data3/private/clsi/AmbigQA/data/ambig_nq/dev_light.json

squad_train=/data3/private/clsi/AmbigQA/data/squad_open/train.json
squad_dev=/data3/private/clsi/AmbigQA/data/squad_open/dev.json
squad_test=/data3/private/clsi/AmbigQA/data/squad_open/test.json

squad_multi_train=/data3/private/clsi/AmbigQA/data/squad_open_multi/train.json
squad_multi_dev=/data3/private/clsi/AmbigQA/data/squad_open_multi/dev.json
squad_multi_test=/data3/private/clsi/AmbigQA/data/squad_open_multi/test.json



# python3 cli.py \
# --seed 12 \
# --do_train \
# --do_predict \
# --task qa \
# --num_train_epochs 4 \
# --dpr_data_dir $dpr_data_dir \
# --output_dir /data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_seed12 \
# --train_file $nq_train \
# --predict_file $nq_dev \
# --bert_name bert-base-uncased \
# --train_M 24 \
# --test_M 10 \
# --train_batch_size 2 \
# --gradient_accumulation_steps 8 \
# --predict_batch_size 32 \
# --eval_period 400 \
# --wait_step 10 \
# --topk_answer 10 \
# --learning_rate 3e-5 \
# --checkpoint /home/sichenglei/bert-base-uncased/pytorch_model.bin
# # --checkpoint ../out/reader_nq_multi_new/best-model.pt





# python3 cli.py \
# --seed 12 \
# --do_predict \
# --task qa \
# --num_train_epochs 3 \
# --dpr_data_dir $dpr_data_dir \
# --output_dir /data3/private/clsi/OpenQA/AmbigQA/out/reader_trivia_new_seed12 \
# --train_file $trivia_train \
# --predict_file $trivia_test \
# --bert_name bert-base-uncased \
# --train_M 24 \
# --test_M 10 \
# --train_batch_size 2 \
# --gradient_accumulation_steps 8 \
# --predict_batch_size 32 \
# --eval_period 800 \
# --wait_step 5 \
# --topk_answer 10 \
# --learning_rate 3e-5 \
# --checkpoint /data3/private/clsi/OpenQA/AmbigQA/out/reader_trivia_new_seed12/best-model.pt

# --checkpoint /home/sichenglei/bert-base-uncased/pytorch_model.bin
# --checkpoint /data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_seed12/best-model.pt
# --checkpoint /home/sichenglei/bert-base-uncased/pytorch_model.bin
# --checkpoint ../out/reader_nq_multi_new/best-model.pt



python3 cli.py \
--seed 12 \
--do_predict \
--task qa \
--num_train_epochs 3 \
--dpr_data_dir $dpr_data_dir \
--output_dir /data3/private/clsi/OpenQA/AmbigQA/out/reader_trivia_multi_new_seed12 \
--train_file $trivia_train \
--predict_file $trivia_multi_test \
--bert_name bert-base-uncased \
--train_M 24 \
--test_M 10 \
--train_batch_size 2 \
--gradient_accumulation_steps 8 \
--predict_batch_size 32 \
--eval_period 800 \
--wait_step 5 \
--topk_answer 10 \
--learning_rate 3e-5 \
--checkpoint /data3/private/clsi/OpenQA/AmbigQA/out/reader_trivia_multi_new_seed12/best-model.pt



# python3 cli.py \
# --seed 100 \
# --do_predict \
# --task qa \
# --num_train_epochs 3 \
# --dpr_data_dir $dpr_data_dir \
# --output_dir /data3/private/clsi/OpenQA/AmbigQA/out/reader_trivia_new_seed100 \
# --train_file $trivia_train \
# --predict_file $trivia_test \
# --bert_name bert-base-uncased \
# --train_M 24 \
# --test_M 10 \
# --train_batch_size 2 \
# --gradient_accumulation_steps 8 \
# --predict_batch_size 32 \
# --eval_period 800 \
# --wait_step 5 \
# --topk_answer 10 \
# --learning_rate 3e-5 \
# --checkpoint /data3/private/clsi/OpenQA/AmbigQA/out/reader_trivia_new_seed100/best-model.pt



python3 cli.py \
--seed 100 \
--do_predict \
--task qa \
--num_train_epochs 3 \
--dpr_data_dir $dpr_data_dir \
--output_dir /data3/private/clsi/OpenQA/AmbigQA/out/reader_trivia_multi_new_seed100 \
--train_file $trivia_train \
--predict_file $trivia_multi_test \
--bert_name bert-base-uncased \
--train_M 24 \
--test_M 10 \
--train_batch_size 2 \
--gradient_accumulation_steps 8 \
--predict_batch_size 32 \
--eval_period 800 \
--wait_step 5 \
--topk_answer 10 \
--learning_rate 3e-5 \
--checkpoint /data3/private/clsi/OpenQA/AmbigQA/out/reader_trivia_multi_new_seed100/best-model.pt



# python3 cli.py \
# --seed 56 \
# --do_predict \
# --task qa \
# --num_train_epochs 3 \
# --dpr_data_dir $dpr_data_dir \
# --output_dir /data3/private/clsi/OpenQA/AmbigQA/out/reader_trivia_new_seed56 \
# --train_file $trivia_train \
# --predict_file $trivia_test \
# --bert_name bert-base-uncased \
# --train_M 24 \
# --test_M 10 \
# --train_batch_size 2 \
# --gradient_accumulation_steps 8 \
# --predict_batch_size 32 \
# --eval_period 800 \
# --wait_step 5 \
# --topk_answer 10 \
# --learning_rate 3e-5 \
# --checkpoint /data3/private/clsi/OpenQA/AmbigQA/out/reader_trivia_new_seed56/best-model.pt



# python3 cli.py \
# --seed 56 \
# --do_predict \
# --task qa \
# --num_train_epochs 3 \
# --dpr_data_dir $dpr_data_dir \
# --output_dir /data3/private/clsi/OpenQA/AmbigQA/out/reader_trivia_multi_new_seed56 \
# --train_file $trivia_train \
# --predict_file $trivia_multi_test \
# --bert_name bert-base-uncased \
# --train_M 24 \
# --test_M 10 \
# --train_batch_size 2 \
# --gradient_accumulation_steps 8 \
# --predict_batch_size 32 \
# --eval_period 800 \
# --wait_step 5 \
# --topk_answer 10 \
# --learning_rate 3e-5 \
# --checkpoint /data3/private/clsi/OpenQA/AmbigQA/out/reader_trivia_multi_new_seed56/best-model.pt


## tmux 110

# python3 cli.py \
#   --do_train \
#   --do_predict \
#   --task qa \
#   --num_train_epochs 4 \
#   --dpr_data_dir $dpr_data_dir \
#   --output_dir ../out/reader_trivia_union_new \
#   --train_file $trivia_union_train \
#   --predict_file $trivia_dev \
#   --bert_name bert-base-uncased \
#   --train_M 24 \
#   --test_M 10 \
#   --train_batch_size 2 \
#   --gradient_accumulation_steps 4 \
#   --predict_batch_size 32 \
#   --eval_period 400 \
#   --wait_step 10 \
#   --topk_answer 10 \
#   --learning_rate 3e-5 


# python3 cli.py \
#   --do_predict \
#   --task qa \
#   --num_train_epochs 4 \
#   --dpr_data_dir $dpr_data_dir \
#   --output_dir ../out/reader_trivia_new \
#   --predict_file $trivia_test \
#   --bert_name bert-base-uncased \
#   --train_M 24 \
#   --test_M 30 \
#   --train_batch_size 2 \
#   --gradient_accumulation_steps 4 \
#   --predict_batch_size 32 \
#   --eval_period 400 \
#   --wait_step 10 \
#   --topk_answer 10 \
#   --learning_rate 3e-5 


# python3 cli.py \
#   --do_predict \
#   --task qa \
#   --num_train_epochs 4 \
#   --dpr_data_dir $dpr_data_dir \
#   --output_dir ../out/reader_trivia_new \
#   --predict_file $trivia_alias_test \
#   --bert_name bert-base-uncased \
#   --train_M 24 \
#   --test_M 30 \
#   --train_batch_size 2 \
#   --gradient_accumulation_steps 4 \
#   --predict_batch_size 32 \
#   --eval_period 400 \
#   --wait_step 10 \
#   --topk_answer 10 \
#   --learning_rate 3e-5 


# python3 cli.py \
#   --do_predict \
#   --task qa \
#   --num_train_epochs 4 \
#   --dpr_data_dir $dpr_data_dir \
#   --output_dir ../out/reader_trivia_new \
#   --predict_file $trivia_multi_test \
#   --bert_name bert-base-uncased \
#   --train_M 24 \
#   --test_M 30 \
#   --train_batch_size 2 \
#   --gradient_accumulation_steps 4 \
#   --predict_batch_size 32 \
#   --eval_period 400 \
#   --wait_step 10 \
#   --topk_answer 10 \
#   --learning_rate 3e-5 


# python3 cli.py \
#   --do_predict \
#   --task qa \
#   --num_train_epochs 4 \
#   --dpr_data_dir $dpr_data_dir \
#   --output_dir ../out/reader_trivia_new \
#   --predict_file $trivia_union_test \
#   --bert_name bert-base-uncased \
#   --train_M 24 \
#   --test_M 30 \
#   --train_batch_size 2 \
#   --gradient_accumulation_steps 4 \
#   --predict_batch_size 32 \
#   --eval_period 400 \
#   --wait_step 10 \
#   --topk_answer 10 \
#   --learning_rate 3e-5 







# python3 cli.py \
#   --do_predict \
#   --task qa \
#   --num_train_epochs 4 \
#   --dpr_data_dir $dpr_data_dir \
#   --output_dir ../out/reader_trivia_new \
#   --predict_file $trivia_test \
#   --bert_name bert-base-uncased \
#   --train_M 24 \
#   --test_M 20 \
#   --train_batch_size 2 \
#   --gradient_accumulation_steps 4 \
#   --predict_batch_size 32 \
#   --eval_period 400 \
#   --wait_step 10 \
#   --topk_answer 10 \
#   --learning_rate 3e-5 


# python3 cli.py \
#   --do_predict \
#   --task qa \
#   --num_train_epochs 4 \
#   --dpr_data_dir $dpr_data_dir \
#   --output_dir ../out/reader_trivia_new \
#   --predict_file $trivia_alias_test \
#   --bert_name bert-base-uncased \
#   --train_M 24 \
#   --test_M 20 \
#   --train_batch_size 2 \
#   --gradient_accumulation_steps 4 \
#   --predict_batch_size 32 \
#   --eval_period 400 \
#   --wait_step 10 \
#   --topk_answer 10 \
#   --learning_rate 3e-5 


# python3 cli.py \
#   --do_predict \
#   --task qa \
#   --num_train_epochs 4 \
#   --dpr_data_dir $dpr_data_dir \
#   --output_dir ../out/reader_trivia_new \
#   --predict_file $trivia_multi_test \
#   --bert_name bert-base-uncased \
#   --train_M 24 \
#   --test_M 20 \
#   --train_batch_size 2 \
#   --gradient_accumulation_steps 4 \
#   --predict_batch_size 32 \
#   --eval_period 400 \
#   --wait_step 10 \
#   --topk_answer 10 \
#   --learning_rate 3e-5 


# python3 cli.py \
#   --do_predict \
#   --task qa \
#   --num_train_epochs 4 \
#   --dpr_data_dir $dpr_data_dir \
#   --output_dir ../out/reader_trivia_new \
#   --predict_file $trivia_union_test \
#   --bert_name bert-base-uncased \
#   --train_M 24 \
#   --test_M 20 \
#   --train_batch_size 2 \
#   --gradient_accumulation_steps 4 \
#   --predict_batch_size 32 \
#   --eval_period 400 \
#   --wait_step 10 \
#   --topk_answer 10 \
#   --learning_rate 3e-5 






# python3 cli.py \
#   --do_predict \
#   --task qa \
#   --num_train_epochs 4 \
#   --dpr_data_dir $dpr_data_dir \
#   --output_dir ../out/reader_trivia_new \
#   --predict_file $trivia_test \
#   --bert_name bert-base-uncased \
#   --train_M 24 \
#   --test_M 10 \
#   --train_batch_size 2 \
#   --gradient_accumulation_steps 4 \
#   --predict_batch_size 32 \
#   --eval_period 400 \
#   --wait_step 10 \
#   --topk_answer 10 \
#   --learning_rate 3e-5 


# python3 cli.py \
#   --do_predict \
#   --task qa \
#   --num_train_epochs 4 \
#   --dpr_data_dir $dpr_data_dir \
#   --output_dir ../out/reader_trivia_new \
#   --predict_file $trivia_alias_test \
#   --bert_name bert-base-uncased \
#   --train_M 24 \
#   --test_M 10 \
#   --train_batch_size 2 \
#   --gradient_accumulation_steps 4 \
#   --predict_batch_size 32 \
#   --eval_period 400 \
#   --wait_step 10 \
#   --topk_answer 10 \
#   --learning_rate 3e-5 


# python3 cli.py \
#   --do_predict \
#   --task qa \
#   --num_train_epochs 4 \
#   --dpr_data_dir $dpr_data_dir \
#   --output_dir ../out/reader_trivia_new \
#   --predict_file $trivia_multi_test \
#   --bert_name bert-base-uncased \
#   --train_M 24 \
#   --test_M 10 \
#   --train_batch_size 2 \
#   --gradient_accumulation_steps 4 \
#   --predict_batch_size 32 \
#   --eval_period 400 \
#   --wait_step 10 \
#   --topk_answer 10 \
#   --learning_rate 3e-5 


# python3 cli.py \
#   --do_predict \
#   --task qa \
#   --num_train_epochs 4 \
#   --dpr_data_dir $dpr_data_dir \
#   --output_dir ../out/reader_trivia_new \
#   --predict_file $trivia_union_test \
#   --bert_name bert-base-uncased \
#   --train_M 24 \
#   --test_M 10 \
#   --train_batch_size 2 \
#   --gradient_accumulation_steps 4 \
#   --predict_batch_size 32 \
#   --eval_period 400 \
#   --wait_step 10 \
#   --topk_answer 10 \
#   --learning_rate 3e-5 

# ## tmux 66
# python3 cli.py \
#   --do_predict \
#   --predict_batch_size 3200 \
#   --task dpr \
#   --dpr_data_dir $dpr_data_dir \
#   --output_dir ../out/reader_squad_new \
#   --predict_file $squad_dev \
#   --bert_name bert-base-uncased \
#   --test_M 100 \
#   --predict_batch_size 3200 \
#   --save_retrieve


# python3 cli.py \
#   --do_predict \
#   --predict_batch_size 3200 \
#   --task dpr \
#   --dpr_data_dir $dpr_data_dir \
#   --output_dir ../out/reader_squad_new \
#   --predict_file $squad_test \
#   --bert_name bert-base-uncased \
#   --test_M 100 \
#   --predict_batch_size 3200 \
#   --save_retrieve


# python3 cli.py \
#   --do_predict \
#   --predict_batch_size 3200 \
#   --task dpr \
#   --dpr_data_dir $dpr_data_dir \
#   --output_dir ../out/reader_squad_new \
#   --predict_file $squad_train \
#   --bert_name bert-base-uncased \
#   --test_M 100 \
#   --predict_batch_size 3200 \
#   --save_retrieve
  

