A biomedical dataset supporting Ontology Enrichment from Texts (OET), by concept discovery and placement, adapting the MedMentions dataset (PubMed abstracts) with SNOMED CT of versions in 2014 and 2017 under the Diseases (disorder) sub-category and the broader categories of Clinical finding, Procedure, and Pharmaceutical / biologic (CPP) product. 

The dataset is available at [Zenodo](https://zenodo.org/record/8043690). 

The dataset consists of two parts:

    MM-S14-Disease      # Diseases (disorder)
    MM-S14-CPP          # Clinical finding, Procedure, and Pharmaceutical / biologic 

Each part of the dataset contains a `mention-level` format and a `mention-edge-pair-level` format. The former has each row as a mention with one or multiple edges, and the latter has each mention-edge pair as a row. For mention-level, the data has `syn_attr` and `syn_full` formats, where the `syn_attr` has synonyms as attributes (or a key in json) and the `syn_full` has each synonym as a row treated as an entitiy for data augmentation only in the training file (train.jsonl). For mention-edge-pair-level sub-folder names, `-in-KB` or `-NIL` means the status of all mentions in the files under the sub-folder; `-complex` means the mentions in the sub-folder are to be placed into a complex edge (i.e., having a complex, direct parent concept, which involves at least one logical operator).

Data folder structure:

```
└───MM-S14-Disease
|   |   mention-level # Mention-level data
|   |   |   syn_attr # synonym as attributes 
|   |   |   |   train.jsonl
|   |   |   |   valid.jsonl
|   |   |   |   test.jsonl     
|   |   |   syn_full # synonym as entities (synonym augmentation, for training data only)
|   |   |   |   train.jsonl
|   |   |   |   valid.jsonl
|   |   |   |   test.jsonl
|   |   mention-edge-pair-level # Mention-edge-pair-level data
|   |   |   |   train.jsonl
|   |   |   |   train-complex.jsonl
|   |   |   |   valid.jsonl
|   |   |   |   valid-complex.jsonl
|   |   |   |   test-in-KB.jsonl
|   |   |   |   test-in-KB-complex.jsonl
|   |   |   |   test-NIL.jsonl
|   |   |   |   test-NIL-complex.jsonl
|   |   ontology # ontology files
|   |   |   SNOMEDCT-US-20140901-Disease_syn_attr_hyp-all.jsonl # entity catalogue, list of jsons.
|   |   |   SNOMEDCT-US-20140901-Disease-edges-atomic.jsonl # edge catalogue (atomic only), list of jsons.
|   |   |   SNOMEDCT-US-20140901-Disease-edges-all.jsonl    # edge catalogue (atomic+complex), list of jsons.
|   |   |   SNOMEDCT-US-20140901-Disease-final.owl # .owl file of the older KB
|   |   |   SNOMEDCT-US-20170301-Disease-final.owl # .owl file of the newer KB
└───MM-S14-CPP
|   |   mention-level # Mention-level data
|   |   |   syn_attr # synonym as attributes
|   |   |   |   train.jsonl
|   |   |   |   valid.jsonl
|   |   |   |   test.jsonl
|   |   |   syn_full # synonym as entities (synonym augmentation, for training data only)
|   |   |   |   train.jsonl
|   |   |   |   valid.jsonl
|   |   |   |   test.jsonl
|   |   mention-edge-pair-level # Mention-edge-pair-level data
|   |   |   |   train.jsonl
|   |   |   |   train-complex.jsonl
|   |   |   |   valid.jsonl
|   |   |   |   valid-complex.jsonl
|   |   |   |   test-in-KB.jsonl
|   |   |   |   test-in-KB-complex.jsonl
|   |   |   |   test-NIL.jsonl
|   |   |   |   test-NIL-complex.jsonl
|   |   ontology
|   |   |   SNOMEDCT-US-20140901-CPP_syn_attr_hyp-all.jsonl # entity catalogue, list of jsons.
|   |   |   SNOMEDCT-US-20140901-CPP-edges-atomic.jsonl # edge catalogue (atomic only), list of jsons.
|   |   |   SNOMEDCT-US-20140901-CPP-edges-all.jsonl    # edge catalogue (atomic+complex), list of jsons.
|   |   |   SNOMEDCT-US-20140901-CPP-final.owl # .owl file of the older KB
|   |   |   SNOMEDCT-US-20170301-CPP-final.owl # .owl file of the newer KB
```

JSON keys for mention-level data:
    
    "context_left"              # left context
    "mention"                   # mention
    "context_right"             # right context 
    "label_concept_UMLS"        # label concept in UMLS
    "label_concept"             # label concept in SNOMED CT older version
    "label_concept_ori"         # label concept original, in SNOMED CT newer version if out of the older version
    "label"                     # entity description
    "parents_concept"           # parent concept SNOMED CT ID (or complex concept IDs)
    "parents"                   # parent concept title
    "children_concept"          # child concept SNOMED CT ID (or complex concept IDs)
    "children"                  # child concept title
    "label_id"                  # row id in the entity catalogue ontology/xxx_syn_attr_hyp-all.jsonl file
    "label_title"               # entity title

JSON keys for mention-edge-pair-level data:

    "context_left"              # left context
    "mention"                   # mention
    "context_right"             # right context 
    "label_concept_UMLS"        # label concept in UMLS
    "label_concept"             # label concept in SNOMED CT older version
    "label_concept_ori"         # label concept original, in SNOMED CT newer version if out of the older version
    "entity_label_id"           # row id in the entity catalogue ontology/xxx_syn_attr_hyp-all.jsonl file
    "entity_label"              # entity description
    "entity_label_title"        # entity title
    "parent_concept"            # parent concept SNOMED CT ID (or complex concept IDs)
    "child_concept"             # child concept SNOMED CT ID (or complex concept IDs)
    "parent"                    # parent concept title
    "child"                     # child concept title
    "edge_label_id"             # row id in the edge catalogue ontology/xxx_edges.jsonl file

Acknowledgement:

    MedMentions dataset (using PubMed Abstracts) https://github.com/chanzuckerberg/MedMentions
    SNOMED CT https://www.nlm.nih.gov/healthit/snomedct/index.html
    UMLS https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html
    DeepOnto library https://github.com/KRR-Oxford/DeepOnto
    snomed-owl-toolkit https://github.com/IHTSDO/snomed-owl-toolkit

p.s. Note that we renamed the sub-folder names after running the scripts for easier understanding.
     The original sub-folder names are below:     
     
        original sub-folder name                    -> new sub-folder name
        st21pv_syn_attr-all-complexEdge-filt        -> mention-level/syn_attr
        st21pv_syn_full-all-complexEdge-filt        -> mention-level/syn_full
        st21pv_syn_attr-all-complexEdge-edges-final -> mention-edge-pair-level
