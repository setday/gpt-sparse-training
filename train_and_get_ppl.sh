python data/shakespeare_char/prepare.py

python train.py config/train_shakespeare_char.py

python train.py config/train_shakespeare_char.py --eval_only=True --init_from='resume'