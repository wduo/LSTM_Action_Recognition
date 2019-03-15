# LSTM action recognition

LSTM action recognition.

# Dependence

```
python2.7
tensorflow
opencv
```

# Quick start

Put imgs into 'data' folder with the follow struct:

```
./data
	--train
	  --ApplyEyeMakeup
	  --ApplyLipstick
	  ...
	  --train_list.txt
	  
	--test(Same as train folder)
	  --ApplyEyeMakeup
	  --ApplyLipstick
	  ...
	  --test_list.txt
	
	data_file.csv
```

run the following command:

```
python s5_train.py
```

# License

LSTM_Action_Recognition is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail at wduo2017[at]163[dot]com. We will send the detail agreement to you.