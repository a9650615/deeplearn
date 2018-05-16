LeNetTensorflow

# install 
 please run
 ```bash
  source ./bin/actiove
 ```
 ```bash
  pip install -r requirements.txt
 ```
 
 if you want to train, you can run 
 ```bash
  python ./tensorflow/network train
 ```
if you want to fetch data for numpy loading, after run train at least once, run
```bash
 python ./tensorflow/network patch
```
the patch data will all in deep_data folder

or if you just want to watch data struct, run 
```bash
 python ./tensorflow/network print
```