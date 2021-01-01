model_dirs = [
    # 'model/spider.bridge.ppl.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201113-113558.vgs0/model-best.16.tar',
    # 'model/spider.bridge.ppl.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201113-113558.ljap/model-best.16.tar',
    # 'model/spider.bridge.ppl.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201113-115517.51x5/model-best.16.tar',
    # 'model/spider.bridge.ppl.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.1-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201112-185010.0rip/model-best.16.tar',
    # 'model/spider.bridge.ppl.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.1-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201112-185458.gryi/model-best.16.tar',
    # 'model/spider.bridge.ppl.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.1-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201112-184443.ss5p/model-best.16.tar'
    # 'model/spider.bridge.ppl.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201128-210933.j6ot/model-best.16.tar',
    # 'model/spider.bridge.ppl.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201128-210906.yi82/model-best.16.tar',
    # 'model/spider.bridge.ppl.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201128-210911.1ngi/model-best.16.tar'
    # 'model/model_analysis/spider.bridge.ppl.2.dn.eo.feat.bert-large-uncased.xavier-1024-384-384-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201216-111910.c73s',
    # 'model/model_analysis/spider.bridge.ppl.2.dn.eo.feat.bert-large-uncased.xavier-1024-384-384-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201216-111910.g965',
    # 'model/model_analysis/spider.bridge.ppl.2.dn.eo.feat.bert-large-uncased.xavier-1024-384-384-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201216-111910.o6jm',
    # 'model/model_analysis/spider.bridge.ppl.2.dn.eo.feat.bert-large-uncased.xavier-1024-384-384-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201216-111913.hptm'
    'model/ensemble_analysis/spider.sqlova.lstm.meta.ts.ppl-0.85.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201218-093155.gjrk/',
    # 70.1, 68.2
    'model/ensemble_analysis/spider.sqlova.lstm.meta.ts.ppl-0.85.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201218-093255.vn2k/',
    # 69.1, 67.1
    # 'model/ensemble_analysis/spider.bridge.lstm.meta.ts.ppl-0.85.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201218-110818.d08k/',
    # 68.4, 67.5
    # 'model/ensemble_analysis/spider.bridge.lstm.meta.ts.ppl-0.85.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201218-093110.a97i/',
    # 68.3, 66.2
    # 'model/ensemble_analysis/spider.bridge.lstm.meta.ts.ppl-0.85.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201218-093153.xf80/',
    # 68.2, 67.6
    # 'model/ensemble_analysis/spider.bridge.lstm.meta.ts.ppl-0.85.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201218-132607.nj9q/',
    # 68.2, 67.4
    # 'model/ensemble_analysis/spider.bridge.lstm.meta.ts.ppl-0.85.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201218-093309.46gg/',
    # 67.7, 67.4
    # 'model/ensemble_analysis/spider.bridge.lstm.meta.ts.ppl-0.85.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201218-093116.8vxb/',
    # 67.2, 67.0
    # 'model/ensemble_analysis/spider.bridge.lstm.meta.ts.ppl-0.85.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201218-093640.ec17/',
    # 67.2, 66.4
    # 'model/ensemble_analysis/spider.sqlova.lstm.meta.ts.ppl-0.85.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201218-093120.3un8/',
    # 66.7, 66.0
    # 'model/ensemble_analysis/spider.bridge.lstm.meta.ts.ppl-0.85.2.dn.eo.feat.bert-large-uncased.xavier-1024-400-400-16-2-0.0005-inv-sqr-0.0005-4000-6e-05-inv-sqr-3e-05-4000-0.3-0.3-0.0-0.0-1-8-0.0-0.0-res-0.2-0.0-ff-0.4-0.0.201218-093133.lmya/',
    # 66.4, 65.8
]