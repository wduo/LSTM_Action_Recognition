# params
TRAIN_TEST_FOLDERS = ['./data/train/', './data/test/']
CATEGORIES = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching']

LSTM_LAYER = 1
FRAME_NUM = 16
HIDDEN_LAYER_NODES = 1024
LEARNING_RATE = 0.000001
BATCH_SIZE = 64

LR_DECAY_STEPS = 1000
LR_DECAY_RATE = 0.8
WEIGHT_DECAY = 0.001
# WEIGHT_DECAY = None
DROPOUT_KEEP_PROB = 0.5
