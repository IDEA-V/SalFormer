nice -5 python trainning_salformer.py --model 'llama' --device 'cuda:1' --kl 10 --cc 5 --nss 2 --lr 0.00006
nice -5 python trainning_salformer.py --model 'bloom' --device 'cuda:2' --kl 10 --cc 5 --nss 2 --lr 0.00006
nice -5 python trainning_salformer.py --model 'bert' --device 'cuda:3' --batch_size 32 --kl 10 --cc 1 --nss 2 --lr 0.00006
