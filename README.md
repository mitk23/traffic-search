# Improving Long-Term Traffic Prediction with Online Search Log Data

### Itsuki Matsunaga, Yuto Kosugi, Ge Hangli, Takashi Michikata, Noboru Koshizuka

## Code
- [experiment.ipynb](https://github.com/mitk23/traffic-search/tree/master/src/experiment.ipynb): description of experiment procedure
- [models.py](https://github.com/mitk23/traffic-search/tree/master/src/models.py): implementation of deep neural networks
- [train.py](https://github.com/mitk23/traffic-search/tree/master/src/train.py): training script
- [predict.py](https://github.com/mitk23/traffic-search/tree/master/src/predict.py): prediction script
- [baseline.py](https://github.com/mitk23/traffic-search/tree/master/src/baseline.py): implementation of baselines
- [transformer.py](https://github.com/mitk23/traffic-search/tree/master/src/transformer.py): implementation of formatting and standardizing tabular data into spatio-temporal dataset

## experiment procedure
see [experiment.ipynb](https://github.com/mitk23/traffic-search/tree/master/src/experiment.ipynb)

## get started
```bash
▶ git clone https://github.com/mitk23/traffic-search.git
▶ cd traffic-search/
▶ docker build -t traffic-search .
▶ docker run --name <container name> -itd \
  --gpus all -p <your jupyter port>:8888 -e CUBLAS_WORKSPACE_CONFIG=:4096:2 \
  --mount type=bind,src=$(pwd),target=$(pwd) -w=$(pwd) \
  traffic-search /bin/bash -c 'jupyter-lab --allow-root --port=8888 --ip=*'
  
▶ docker exec -it <container name> /bin/bash
# container
$ cd src
$ python3.9 train.py -h
```
and see [experiment.ipynb](https://github.com/mitk23/traffic-search/tree/master/src/experiment.ipynb)
