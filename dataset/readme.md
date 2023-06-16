A biomedical dataset supporting Ontology Enrichment from Texts (OET), by concept discovery and placement, adapting the MedMentions dataset (PubMed abstracts) with SNOMED CT of versions in 2014 and 2017 under the Diseases (disorder) sub-category and the broader categories of Clinical finding, Procedure, and Pharmaceutical / biologic (CPP) product. 

The dataset is available at [Zenodo](https://zenodo.org/record/8043690). 

The dataset consists of two parts:

    MM-S14-Disease      # Diseases (disorder)
    MM-S14-CPP          # Clinical finding, Procedure, and Pharmaceutical / biologic 

Data folder structure:

└───MM-S14-Disease
|   |   mention-level # Mention-level data
|   |   |   syn_attr # synonym as attributes 
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
|   |   |   SNOMEDCT-US-20140901-Disease_syn_attr_hyp-all.jsonl # entity catalogue, list of jsons, each is an entity.
|   |   |   SNOMEDCT-US-20140901-Disease-edges.jsonl # list of jsons, each is an edge.
|   |   |   SNOMEDCT-US-20140901-Disease-final.owl # .owl file of the older KB
|   |   |   SNOMEDCT-US-20170301-Disease-final.owl # .owl file of the newer KB
└───MM-S14-CPP
|   |   mention-level # Mention-level data
|   |   |   syn_attr # synonym as attributes
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
|   |   |   SNOMEDCT-US-20140901-CPP_syn_attr_hyp-all.jsonl # entity catalogue, list of jsons, each is an entity.
|   |   |   SNOMEDCT-US-20140901-CPP-edges.jsonl # edge catalogue, list of jsons, each is an edge.
|   |   |   SNOMEDCT-US-20140901-CPP-final.owl # .owl file of the older KB
|   |   |   SNOMEDCT-US-20170301-CPP-final.owl # .owl file of the newer KB

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
    "childs"                    # child concept title
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