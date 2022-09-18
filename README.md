For trainig check config.json and choose necessary model/optimizer and so on.
python train.py -c config.json

Use tensorboard for control learning process.
tensorboard --logdir "path_to_your_log_dir"

For validation use your checkpoint and pass it with config from saved directory, do not forget to change path for train data to validation datapath.
python test.py --resume "path_to_your_checkpoint"
