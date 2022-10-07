class Config:
    BATCH_SIZE=2
    MAX_STEPS=10
    INIT_LR=0.0001
    ADAM_WEIGHT_DECAY=1e-8

    DATA_ROOT_PATH='data'
    ID3_PRETRAINED_WEIGHTS_PATH='weights/pretrained_imagenet/rgb_imagenet.pt'
    SAVE_MODEL_PATH='weights/inception3d/'
    TRAIN_SPLIT_FILE='preprocess/nslt_2000.json'
