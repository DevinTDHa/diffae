#!/bin/bash

python /home/tha/diffae/counterfactual/diffeo_cf.py \
    --image_path=/home/tha/diffae/imgs_align/sandy.png \
    --resize=256 \
    --save_at=0.8 \
    --rmodel_path="/home/tha/master-thesis-xai/thesis_utils/scripts/train/runs/imdb_clean_256/version_1/checkpoints/epoch=49-step=287350.ckpt" \
    --rmodel_type="resnet152"
