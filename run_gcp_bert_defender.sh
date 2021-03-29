#!/bin/bash

# conda create --name bert_defender_env
conda activate bert_defender_env
# conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
# conda install -c conda-forge hnswlib
# conda install numpy
# conda install -c conda-forge tqdm
# conda install -c anaconda boto3
# conda install -c anaconda requests
# conda install -c conda-forge scikit-learn
# conda install -c anaconda nltk
# python -m nltk.downloader punkt
# conda activate base
# source /opt/conda/bin/activate
# conda deactivate
# python3 -m venv .bert_defender_env
# source ~/.bert_defender_env/bin/activate

# Install dependencies and speechbrain
# cd ~/bert-defender/
# pip install -r requirements.txt
# pip install -e .
# pip install tensorboard
# pip install matplotlib
# pip install pandas
# pip install seaborn
# cd ~

# ./bert-defender/run_gcp_bert_defender.sh |& tee >> ./bert-defender/exp_out.txt &
# run_gcp_bert_defender.sh |& tee >> exp_out.txt &
# ps -ef | grep monitor_gpu
# ps -ef | grep tensorboard
# ps -ef | grep run_gcp_voicebank_enhance

# ./bert-defender/monitor_gpu.sh &

# python ./bert_discriminator.py \
# --task_name sst-2 \
# --do_train  \
# --do_lower_case   \
# --data_dir data/sst-2/train.tsv   \
# --bert_model bert-base-uncased   \
# --max_seq_length 128   \
# --train_batch_size 8   \
# --learning_rate 2e-5   \
# --num_train_epochs 25 \
# --output_dir ./tmp/disc/

# python ./bert_generator.py \
# --task_name sst-2 \
# --do_train  \
# --do_lower_case \
# --data_dir data/SST-2/ \
# --bert_model bert-base-uncased  \
# --max_seq_length 64   \
# --train_batch_size 8  \
# --learning_rate 2e-5   \
# --num_train_epochs 25 \
# --output_dir ./tmp/gnrt/

python bert_discriminator.py \
--task_name sst-2 \
--do_eval \
--eval_batch_size 32 \
--do_lower_case \
--data_dir data/SST-2/add_1/ \
--data_file data/SST-2/add_1/test.tsv \
--bert_model bert-base-uncased   \
--max_seq_length 128  \
--train_batch_size 16   \
--learning_rate 2e-5   \
--num_eval_epochs 5 \
--output_dir models/ \
--single  \

# python bert_generator.py 
# --task_name sst-2 
# --do_eval  
# --do_lower_case   
# --data_dir data/SST-2/add_1/  
# --bert_model bert-base-uncased   
# --max_seq_length 64   
# --train_batch_size 8  
# --learning_rate 2e-5   
# --output_dir ./tmp/sst2-gnrt/ 
# --num_eval_epochs 2


cd ~

