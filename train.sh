CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch  \
train_anchor.py --cfg ./config/train.yaml