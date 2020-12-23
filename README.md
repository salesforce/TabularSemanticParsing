# Bridging Textual and Tabular Data for Cross-Domain Text-to-SQL Semantic Parsing

This repository contains the source code release of the paper: [Bridging Textual and Tabular Data for Cross-Domain Text-to-SQL Semantic Parsing](http://victorialin.net/pubs/bridge-emnlp2020.pdf). Xi Victoria Lin, Richard Socher and Caiming Xiong. Findings of EMNLP 2020.

## Overview

Cross-Domain tabular semantic parsing is the task of predicting the executable target query language given a natural language question issued to some database. The model may or may not have been trained on the target database.

This library implements 
- A strong neural cross-domain text-to-SQL semantic parser that achieved state-of-the-art performance on both the [Spider](https://yale-lily.github.io/spider) and [WikiSQL](https://github.com/salesforce/WikiSQL) benchmark. The parser can be adapted to learn the mapping from text to other DB query languages such as SOQL and Arklang by redefining its target language.
- A set of [SQL processing tools](moz_sp) for parsing, tokenizing and validating SQL queries, adapted from the [Moz SQL Parser](https://github.com/mozilla/moz-sql-parser).

## Model

![BRIDGE architecture](http://victorialin.net/img/bridge_architecture.png)

The model takes a natural language utterance and a database (schema + field picklists) as input, and output an executable SQL query.
- **Preprocessing:** We concatenate the serialized database schema with the utterance to form a tagged sequence. A [fuzzy string matching algorithm](src/common/content_encoder.py) is used to find the picklist items mentioned in the utterance. The mentioned picklist items are appended to the corresponding field name in the tagged sequence.
- **Translating:** The hybrid sequence is passed through the BRIDGE model, which output raw program sequences with probabitlity scores via beam search.
- **Postprocessing:** The raw program sequences are passed through a SQL checker, which verifies the program syntax and schema consistency. Sequences that failed to pass the checker are discarded from the output.

## Quick Start

### Install Dependencies

Our code has been tested with Pytorch 1.7 and Cuda 11 with a single GPU.
```
pip install torch torchvision
python3 -m pip install -r requirements.txt
```

### Set up environment
```
export PYTHONPATH = `pwd` && mkdir data
```

### Process Data

Spider

Download [spider.zip](https://drive.google.com/u/1/uc?export=download&confirm=pft3&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0), unzip folder, and manually merge `spider/train_spider.json` with `spider/train_others.json` into a single file `spider/train.json`.
```
mv spider data/ 
./experiment-bridge.sh configs/bridge/spider-bridge-bert-large.sh --process_data 0
```

WikiSQL

Download [data.tar.bz2](https://github.com/salesforce/WikiSQL/blob/master/data.tar.bz2).
```
tar xf data.tar.bz2 -C data && mv data/data data/wikisql1.1
./experiment-brdige.sh configs/bridge/wikisql-bridge-bert-large.sh --process_data 0
```

The processed data will be stored in a separate pickle file. 

### Train 
Train the model.

Spider
```
./experiment-bridge.sh configs/bridge/spider-bridge-bert-large.sh --train 0
```

WikiSQL
```
./experiment-brdige.sh configs/bridge/wikisql-bridge-bert-large.sh --train 0
```

### Inference
Decode SQL predictions from pre-trained models. 

Spider
```
./experiment-bridge.sh configs/bridge/spider-bridge-bert-large.sh --inference 0
```

WikiSQL
```
./experiment-brdige.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0
```

**Note:** Evaluation metrics will be printed out at the end of decoding. The Spider inference takes some time as the static SQL checker causes the delay. The WikiSQL evaluation takes some time because it computes execution accuracy.

### Inference
Decoding with model ensembles. First, set the checkpoint directories of the models to be combined in [experiment.py](src/experiments.py#L143).

Spider
```
./experiment-bridge.sh configs/bridge/spider-bridge-bert-large.sh --ensemble_inference 0
```

## Citation
If you find the resource in this repository helpful, please cite
```
@inproceedings{LinRX2020:BRIDGE, 
  author = {Xi Victoria Lin and Richard Socher and Caiming Xiong}, 
  title = {Multi-Hop Knowledge Graph Reasoning with Reward Shaping}, 
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural
               Language Processing: Findings, {EMNLP} 2020, November 16-20, 2020},
  year = {2020} 
}
```

## Related Links
The parser has been integrated in the Photon web demo: http://naturalsql.com/. Please visit our website to test it live and try it on your own databases!
