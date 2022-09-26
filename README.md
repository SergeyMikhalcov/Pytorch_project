For trainig check config.json and choose necessary model/optimizer and so on.

Use tensorboard for control learning process.
tensorboard --logdir "path_to_your_log_dir"

For validation use your checkpoint and pass it with config from saved directory, do not forget to change path for train data to validation datapath.
python test.py --resume "path_to_your_checkpoint"

a) 
    1) Caesar training:
       python train.py -c config_rnn_caesar.json

    2) Caesar validation/inference: 
       python test.py -c saved_checkpoints\models\RNN_Caesar\0923_152755\config.json -r saved\models\RNN_Caesar\0923_152755\model_best.pth

b) 
    1) Sequence training:
        python train.py -c config_for_seq.json

    2) Sequence validation/inference:
        python test.py -c saved_checkpoints\models\Seq_LSTM\0923_154333\config.json -r saved\models\Seq_LSTM\0923_154333\model_best.pth
