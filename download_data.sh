data_dir=/data3/private/clsi/AmbigQA
# dpr_data_dir=/data3/private/clsi/DPR
dpr_data_dir=/data3/private/clsi/DPR/data/wikipedia_split

# python3 download_data.py --resource data.wikipedia_split.psgs_w100 --output_dir ${data_dir} # provided by original DPR
# python3 download_data.py --resource data.wikipedia_split.psgs_w100_20200201 --output_dir ${data_dir} # only for AmbigQA
# python3 download_data.py --resource data.nqopen --output_dir ${data_dir}
# python3 download_data.py --resource data.ambigqa --output_dir ${data_dir}
python3 download_data.py --resource data.wikipedia_split.psgs_w100 --output_dir ${dpr_data_dir}


# python3 download_data.py --resource checkpoint.retriever.multiset.bert-base-encoder --output_dir ${dpr_data_dir}
