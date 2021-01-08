# Untranslatable Question Detection for Cross-Domain Text-to-SQL

This directory contains the implementation of the question translatability checker described in section 3.2.1 of the paper 
[Photon: A Robust Cross-Domain Text-to-SQL System](https://arxiv.org/abs/2007.15280). Zeng et al. ACL System Demonstration 2020.

## QuickStart

### Download UTran-SQL (Spider) Dataset

Download the Spider section of the UTran-SQL dataset [here](https://drive.google.com/file/d/1gKqaEqbegPRzFt2lDigJiKG7bPZU6uU7/view?usp=sharing) and place it under the data directory.
```
unzip spider_ut.zip
mv spider_ut.zip ../../data
```

### Train Untranslatable Question Detector
```
python3 trans_checker.py --train --gpu 0
```

### Inference
```
python3 trans_checker.py --inference --gpu 0
```

### Hyperparameter Changes
You can change the hyperparameters of the model in the [configuration file](args.py).

## Citation
If you find the code helpful, please cite
```
@inproceedings{DBLP:conf/emnlp/LinSX20,
  author    = {Xi Victoria Lin and
               Richard Socher and
               Caiming Xiong},
  editor    = {Trevor Cohn and
               Yulan He and
               Yang Liu},
  title     = {Bridging Textual and Tabular Data for Cross-Domain Text-to-SQL Semantic
               Parsing},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural
               Language Processing: Findings, {EMNLP} 2020, Online Event, 16-20 November
               2020},
  pages     = {4870--4888},
  publisher = {Association for Computational Linguistics},
  year      = {2020},
  url       = {https://www.aclweb.org/anthology/2020.findings-emnlp.438/},
  timestamp = {Thu, 12 Nov 2020 17:18:16 +0100},
  biburl    = {https://dblp.org/rec/conf/emnlp/LinSX20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
