nice -5 python evaluation.py --model 'bert' --ckpt './ckpt/model_bert_freeze_10kl_5cc_2nss.tar' --device 'cuda:0'
nice -5 python evaluation.py --model 'bloom' --ckpt './ckpt/model_bloom_freeze_10kl_5cc_2nss.tar' --device 'cuda:0'
nice -5 python evaluation.py --model 'llama' --batch_size 16 --ckpt './ckpt/model_llama_freeze_10kl_5cc_2nss.tar' --device 'cpu'