CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python inference_predict.py --cfg ./config/vq2d_all_transformer2_anchor_dinov2_inference.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python inference_predict.py --cfg ./config/vq2d_all_transformer2_anchor_dinov2_inference.yaml --eval