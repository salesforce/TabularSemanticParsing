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
@inproceedings{DBLP:conf/acl/ZengLHSXLK20,
  author    = {Jichuan Zeng and
               Xi Victoria Lin and
               Steven C. H. Hoi and
               Richard Socher and
               Caiming Xiong and
               Michael R. Lyu and
               Irwin King},
  editor    = {Asli {\c{C}}elikyilmaz and
               Tsung{-}Hsien Wen},
  title     = {Photon: {A} Robust Cross-Domain Text-to-SQL System},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational
               Linguistics: System Demonstrations, {ACL} 2020, Online, July 5-10,
               2020},
  pages     = {204--214},
  publisher = {Association for Computational Linguistics},
  year      = {2020},
  url       = {https://doi.org/10.18653/v1/2020.acl-demos.24},
  doi       = {10.18653/v1/2020.acl-demos.24},
  timestamp = {Fri, 08 Jan 2021 21:20:19 +0100},
  biburl    = {https://dblp.org/rec/conf/acl/ZengLHSXLK20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
