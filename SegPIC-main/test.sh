CUDA_VISIBLE_DEVICES=1 python -m compressai.utils.eval_model \
    -d /opt/data/private/dataset/Kodak \
    -a segpic \
    --cuda \
    -p /opt/data/private/ckpt/segpic_opensource/segpic_0018_best.pth.tar \
    --testNoMask \
    --grid 4 \

