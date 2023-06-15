# OET
Ontology Enrichment from Texts (OET): A Biomedical Dataset for Concept Discovery and Placement

The repository provides scripts for data creation, with guideline to implement baseline methods for out-of-KB mention discovery and concept placement.

# Dataset link 
The dataset is available at [Zenodo](https://zenodo.org/record/8043690). 

# Data and processing sources
Before data creation, the below sources need to be downloaded.
* SNOMED CT https://www.nlm.nih.gov/healthit/snomedct/archive.html (and use snomed-owl-toolkit to form .owl files)
* UMLS https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html (and mainly use MRCONSO for mapping UMLS to SNOMED CT)
* MedMentions https://github.com/chanzuckerberg/MedMentions (source of entity linking)

The below tools and libraries are used.
* Protege http://protegeproject.github.io/protege/
* snomed-owl-toolkit https://github.com/IHTSDO/snomed-owl-toolkit
* DeepOnto https://github.com/KRR-Oxford/DeepOnto (based on OWLAPI https://owlapi.sourceforge.net/) for ontology processing and complex concept verbalisation

# Data creation scripts
The data creation scripts are available at `preprocessing` folder, where `run_preprocess_ents_and_data+new.sh` provides an overall shell script that calls the other `.py` files.

# Methods
## Out-of-KB mention discovery
We used BLINKout with default parameters and the value of k as 50.

## Concept placement
We used an edge-Bi-encoder, which adapts the original BLINK/BLINKout model by matching a mention to an edge `<parent, child>`.

Then after selecting top-k edges, an optional step is to choose the correct ones for the evaluation. We tested GPT-3.5 (gpt-3.5-turbo) via OpenAI API. Details of the prompt and implementation are available in `method/concept-placement` folder.

# Acknowledgement
* The baseline implementations are based on [BLINKout paper](https://arxiv.org/abs/2302.07189) and [BLINK repository](https://github.com/facebookresearch/BLINK) under the MIT liscence. 
* The zero-shot prompting uses [GPT-3.5](https://platform.openai.com/docs/models) from OpenAI API.
* Acknowledgement to all [data and processing sources](https://github.com/KRR-Oxford/OET#data-and-processing-sources) listed above.