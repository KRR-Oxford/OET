from deeponto import init_jvm
init_jvm("32g")
from deeponto.onto import Ontology, OntologyReasoner, OntologyVerbaliser, OntologySyntaxParser
onto_parser = OntologySyntaxParser()
import re
CONST_NULL_NODE = "SCTID_NULL" # null node
CONST_THING_NODE = "SCTID_THING" # Thing node

def load_SNOMEDCT_deeponto(onto_owl_fn):
    print('loading ontology:',onto_owl_fn)
    onto_sno = Ontology(onto_owl_fn)
    return onto_sno

def load_SNOMEDCT_owl2dict_ids(onto_owl_fn):
    #onto_sno = Ontology(onto_owl_fn)
    onto_sno = load_SNOMEDCT_deeponto(onto_owl_fn)
    dict_SCTID_onto = onto_sno.owl_classes
    print('num of %s IDs:' % onto_owl_fn,len(dict_SCTID_onto))
    print('First 5 IDs:', list(dict_SCTID_onto.items())[:5]) # each is of format ('http://snomed.info/id/10001005', <java object 'uk.ac.manchester.cs.owl.owlapi.OWLClassImpl'>)
    return dict_SCTID_onto

def load_deeponto_reasoner(onto_sno):
    print('loading ontology reasoner.')
    onto_sno_reasoner = OntologyReasoner(onto_sno)
    return onto_sno_reasoner

def load_deeponto_verbaliser(onto_sno):
    print('loading ontology verbaliser.')
    onto_sno_verbaliser = OntologyVerbaliser(onto_sno)
    return onto_sno_verbaliser

def deeponto2dict_ids(onto):
    #onto_sno = Ontology(onto_owl_fn)
    #onto_sno = load_SNOMEDCT_deeponto(onto_owl_fn)
    dict_SCTID_onto = onto.owl_classes
    print('num of IDs:',len(dict_SCTID_onto))
    print('First 5 IDs:', list(dict_SCTID_onto.items())[:5]) # each is of format ('http://snomed.info/id/10001005', <java object 'uk.ac.manchester.cs.owl.owlapi.OWLClassImpl'>)
    return dict_SCTID_onto

def deeponto2dict_ids_obj_prop(onto):
    dict_SCTID_onto = onto.owl_object_properties
    print('num of IDs:',len(dict_SCTID_onto))
    print('First 5 IDs:', list(dict_SCTID_onto.items())[:5]) # each is of format ('http://snomed.info/id/10001005', <java object 'uk.ac.manchester.cs.owl.owlapi.OWLClassImpl'>)
    return dict_SCTID_onto

def deeponto2dict_ids_ann_prop(onto):
    dict_SCTID_onto = onto.owl_annotation_properties
    print('num of IDs:',len(dict_SCTID_onto))
    print('First 5 IDs:', list(dict_SCTID_onto.items())[:5]) # each is of format ('http://snomed.info/id/10001005', <java object 'uk.ac.manchester.cs.owl.owlapi.OWLClassImpl'>)
    return dict_SCTID_onto

def get_definition_in_onto_from_iri(onto,iri):
    return onto.get_owl_object_annotations(iri,annotation_property_iri='http://www.w3.org/2004/02/skos/core#definition')

def get_rdfslabel_in_onto_from_iri(onto,iri):
    return onto.get_owl_object_annotations(iri,annotation_property_iri='http://www.w3.org/2000/01/rdf-schema#label')

def get_preflabel_in_onto_from_iri(onto,iri):
    return onto.get_owl_object_annotations(iri,annotation_property_iri='http://www.w3.org/2004/02/skos/core#prefLabel')

def get_altlabel_in_onto_from_iri(onto,iri):
    return onto.get_owl_object_annotations(iri,annotation_property_iri='http://www.w3.org/2004/02/skos/core#altLabel')

def get_title_in_onto_from_iri(onto,iri):
    set_rdfslabels = get_rdfslabel_in_onto_from_iri(onto,iri)
    if len(set_rdfslabels) > 0:
        concept_tit = list(set_rdfslabels)[0]
    else:
        concept_tit = ''
    return concept_tit

def parse_owl_class_expression(owlClassExpression,onto_parser=OntologySyntaxParser()):
    '''
    return the parsed id string of a (complex) concept
    '''
    return onto_parser.parse(owlClassExpression).children[0]

def clean_id_in_parsed_owl_class_expression(parsed_owlClassExpression_str,prefix="http://snomed.info/id/"):
    if prefix != "":
        parsed_owlClassExpression_str = parsed_owlClassExpression_str.replace(prefix,"")
    return parsed_owlClassExpression_str

def verbalise_concept(onto_verbaliser,parsed_owlClassExpression):
    '''
    return the CfgNode output of the verbalisation (see https://github.com/KRR-Oxford/DeepOnto/blob/main/src/deeponto/onto/verbalisation.py)

    (CfgNode): A nested dictionary that presents the details of verbalisation. The verbalised string can be accessed with the key `["verbal"]`.
    CfgNode(
        {
            "verbal": verbal,
            "property": object_property,
            "class": class_expression,
            "type": xxx (IRI, NEG, etc.)
        }
    '''
    verbalised_CfgNode = onto_verbaliser.verbalise_class_expression(parsed_owlClassExpression)
    #print('verbalised_CfgNode:',verbalised_CfgNode)
    return verbalised_CfgNode

# def verbalise_concept_CfgNode_ids(verbalised_CfgNode):
#     '''get id str representation of a (complex) concept'''
#     assert 'class' in verbalised_CfgNode or 'classes' in verbalised_CfgNode
#     if 'class' in verbalised_CfgNode:
#         verbalised_ids_str = verbalised_CfgNode["class"]
#     elif 'classes' in verbalised_CfgNode:
#         verbalised_ids_list = []
#         for atomic_cfgNode in verbalised_CfgNode["classes"]:
#             verbalised_ids_list.append(get_SCTID_id_from_iri(atomic_cfgNode['iri']))
#         verbalised_ids_str = ' '.join(verbalised_ids_list)
#     return verbalised_ids_str

def verbalise_concept_CfgNode(verbalised_CfgNode):
    '''get verbalised str representation of a (complex) concept'''
    verbalised_concept_str = verbalised_CfgNode["verbal"]
    return verbalised_concept_str

def check_subsumption(onto_reasoner,sub_entity,super_entity):
    '''
        The sub_entity and super_entity are OWLObject.
    '''
    return onto_reasoner.check_subsumption(sub_entity,super_entity)

def check_leaf(onto,dict_SCTID_onto,concept_iri):
    concept_iri_obj = dict_SCTID_onto[concept_iri]
    set_children = onto.get_asserted_children(concept_iri_obj)
    #print(concept_iri,'children:',set_children)
    return len(set_children) == 0

def _sort_list(lst):
    '''
    sort the list ascendingly
    '''
    return sorted(lst)
    
def get_entity_info(onto,concept_iri, sorting=False):
    '''
    get entity name (concept_tit), entity definition, and entity synonyms
    TODO: the order of output, especially concept_syns is not fixed. Probably relevant to onto.get_owl_object_annotations().
    '''
    set_rdfslabels = get_rdfslabel_in_onto_from_iri(onto,concept_iri)
    set_preflabels = get_preflabel_in_onto_from_iri(onto,concept_iri)
    set_altlabels = get_altlabel_in_onto_from_iri(onto,concept_iri)
    set_definitions = get_definition_in_onto_from_iri(onto,concept_iri)
    #print('set_preflabels:',set_preflabels)
    #print('set_altlabels:',set_altlabels)
    #print('set_definitions:',set_definitions)

    lst_rdfslabels = _sort_list(list(set_rdfslabels)) if sorting else list(set_rdfslabels)
    lst_preflabels = _sort_list(list(set_preflabels)) if sorting else list(set_preflabels)
    lst_altlabels = _sort_list(list(set_altlabels)) if sorting else list(set_altlabels)
    lst_definitions = _sort_list(list(set_definitions)) if sorting else list(set_definitions)

    if len(lst_definitions) > 0:
        concept_def = lst_definitions[0] # only use the first definition given that they are so similar to each other
    else:
        concept_def = ''    
    if len(lst_rdfslabels) > 0:
        concept_tit = lst_rdfslabels[0]
    else:
        concept_tit = ''
    if len(lst_rdfslabels) > 1:
        concept_syns = '|'.join(lst_rdfslabels[1:] + lst_preflabels + lst_altlabels)
    else:
        concept_syns = '|'.join(lst_preflabels + lst_altlabels)
    #print('list(set_preflabels):',list(set_preflabels))    
    #print('list(set_altlabels):',list(set_altlabels))
    #print('concept_syns:',concept_syns)
    return concept_tit,concept_def,concept_syns

def get_entity_id_and_tit_complex(onto_verbaliser,owl_class_expression):
    class_rangeNode = parse_owl_class_expression(owl_class_expression,onto_parser)
    complex_concept_id = class_rangeNode.text
    complex_concept_tit = verbalise_concept_CfgNode(verbalise_concept(onto_verbaliser, class_rangeNode))
    return complex_concept_id,complex_concept_tit

def get_complex_entities(onto):
    return onto.get_asserted_complex_classes(gci_only=True)

def get_SCTID_from_OWLobj(OWLobj):
    return get_SCTID_id_from_iri(str(OWLobj.getIRI()))

def get_iri_from_SCTID_id(SCTID,postfix='http://snomed.info/id/'):
    return postfix+SCTID

def get_SCTID_id_from_iri(iri,postfix='http://snomed.info/id/'):
    assert iri.startswith(postfix)
    return iri[len(postfix):]

def _extract_iris_in_parsed_complex_concept(parsed_complex_str):
    '''
    extract the list of iris from a parsed complex concept in string form 
    '''
    pattern = "<(.*?)>"
    list_iris = re.findall(pattern,parsed_complex_str)
    return list_iris

def get_in_KB_direct_children(onto,concept_iri,dict_SCTID_onto,dict_SCTID_onto_filtering,onto_verbaliser,concept_type="atomic"):
    '''return a direct list of parent ids, which are in the old ontology.
        concept_type: "atomic", "complex", "all" (or any other strings)

        Note: concept_iri is the string of atomic concept, and can also be the owl_class_expression for a complex concept. 
    '''
    # if concept type is "all", then return both the "atomic" and "complex" ones
    if concept_type == "all":
        list_children_new_final_ = []
        for concept_type_ in ["atomic", "complex"]:
            list_children_new_ = get_in_KB_direct_children(onto,concept_iri,dict_SCTID_onto,dict_SCTID_onto_filtering,onto_verbaliser,concept_type=concept_type_)
            list_children_new_final_ = list_children_new_final_ + list_children_new_
        return list_children_new_final_
    
    def get_in_KB_direct_children_recur(onto,concept_iri,dict_SCTID_onto,dict_SCTID_onto_filtering,onto_verbaliser,concept_type=concept_type): 
        if concept_iri in dict_SCTID_onto:
            # atomic concept
            concept_iri_obj = dict_SCTID_onto[concept_iri]
            set_children = onto.get_asserted_children(concept_iri_obj)
        else:
            # complex concept 
            concept_iri_obj = concept_iri
            set_children = {} # so far, setting a complex concept's child as an empty set 
        if concept_type == "atomic":
            # only atomic        
            set_children = {child for child in set_children if OntologyReasoner.has_iri(child)}
            list_children = [get_SCTID_from_OWLobj(child) for child in set_children]
            #print('list_children:',list_children)
            list_children_new = list_children[:]    
            if dict_SCTID_onto_filtering:
                for child in list_children: 
                    child_iri = get_iri_from_SCTID_id(child)
                    # filtering and recursive calling till getting the in-KB one
                    if not child_iri in dict_SCTID_onto_filtering:
                        print('children %s of %s not in KB, fetch next' % (child_iri,concept_iri))
                        list_children_ = get_in_KB_direct_children_recur(onto,child_iri,dict_SCTID_onto,dict_SCTID_onto_filtering,onto_verbaliser,concept_type=concept_type)
                        list_children_new = list(dict.fromkeys(list_children_new + list_children_))
                        list_children_new.remove(child)
                        print('list_children_:',list_children_)
                        print('list_children_new:',list_children_new)
        elif concept_type == "complex":
            # for complex concepts, only go one level, no recur
            set_children = {child for child in set_children if not OntologyReasoner.has_iri(child)}
            assert onto_verbaliser != None
            list_children = [parse_owl_class_expression(child,onto_parser) for child in set_children]
            list_children_new = list_children[:]
            # filter the one with attributes or classes not in-KB
            if dict_SCTID_onto_filtering:
                #print('start filtering complex children')
                for child in list_children: 
                    list_iris_in_child = _extract_iris_in_parsed_complex_concept(child.text)
                    print("list_iris_in_child:",list_iris_in_child)
                    for iri_in_child in list_iris_in_child:
                        if not iri_in_child in dict_SCTID_onto_filtering:
                            list_children_new.remove(child)
                            print('list_children_new:',list_children_new, 'child', child, 'removed')
                            break            
        return list_children_new
    #print('start')
    list_children_new_final = get_in_KB_direct_children_recur(onto,concept_iri,dict_SCTID_onto,dict_SCTID_onto_filtering,onto_verbaliser,concept_type=concept_type)
    #print('list_children_new_final:',list_children_new_final)
    #print('finish')
    return list_children_new_final
    
def get_in_KB_direct_parents(onto,concept_iri,dict_SCTID_onto,dict_SCTID_onto_filtering,onto_verbaliser,concept_type="atomic"):  
    '''return a direct list of parent ids, which are in the old ontology.
        concept_type: "atomic", "complex", "all" (or any other strings)
       
       Note: concept_iri is the string of atomic concept, and can also be the owl_class_expression for a complex concept.
    '''
    # if concept type is "all", then return both the "atomic" and "complex" ones
    if concept_type == "all":
        list_parents_new_final_ = []
        for concept_type_ in ["atomic", "complex"]:
            list_parents_new_ = get_in_KB_direct_parents(onto,concept_iri,dict_SCTID_onto,dict_SCTID_onto_filtering,onto_verbaliser,concept_type=concept_type_)
            list_parents_new_final_ = list_parents_new_final_ + list_parents_new_
        return list_parents_new_final_

    def get_in_KB_direct_parents_recur(onto,concept_iri,dict_SCTID_onto,dict_SCTID_onto_filtering,onto_verbaliser,concept_type=concept_type):
        if concept_iri in dict_SCTID_onto:
            # atomic concept
            concept_iri_obj = dict_SCTID_onto[concept_iri]
            set_parents = onto.get_asserted_parents(concept_iri_obj)
        
        else:
            # complex concept 
            concept_iri_obj = concept_iri
            set_parents = {} # so far, setting a complex concept's parent as an empty set 
        #print("concept_iri_obj:",concept_iri_obj)
        if concept_type == "atomic":
            # only atomic        
            set_parents = {parent for parent in set_parents if OntologyReasoner.has_iri(parent)}
            list_parents = [get_SCTID_from_OWLobj(parent) for parent in set_parents]
            #print('list_parents:',list_parents)
            list_parents_new = list_parents[:]
            if dict_SCTID_onto_filtering:
                for parent in list_parents: 
                    parent_iri = get_iri_from_SCTID_id(parent)
                    # filtering and recursive calling till getting the in-KB one
                    if not parent_iri in dict_SCTID_onto_filtering:
                        print('parent %s of %s not in KB, fetch next' % (parent_iri,concept_iri))
                        list_parents_ = get_in_KB_direct_parents_recur(onto,parent_iri,dict_SCTID_onto,dict_SCTID_onto_filtering,onto_verbaliser,concept_type=concept_type)
                        list_parents_new = list(dict.fromkeys(list_parents_new + list_parents_))
                        list_parents_new.remove(parent)
                        print('list_parents_:',list_parents_)
                        print('list_parents_new:',list_parents_new)
        elif concept_type == "complex":
            # for complex concepts, only go one level, no recur
            set_parents = {parent for parent in set_parents if not OntologyReasoner.has_iri(parent)}
            #list_parents_new = list(set_parents)
            assert onto_verbaliser != None
            list_parents = [parse_owl_class_expression(parent,onto_parser) for parent in set_parents]
            list_parents_new = list_parents[:]
            # filter the one with attributes or classes not in-KB
            if dict_SCTID_onto_filtering:
                #print('start filtering complex parent')
                for parent in list_parents: 
                    list_iris_in_parent = _extract_iris_in_parsed_complex_concept(parent.text)
                    print("list_iris_in_parent:",list_iris_in_parent)
                    for iri_in_parent in list_iris_in_parent:
                        if not iri_in_parent in dict_SCTID_onto_filtering:
                            # only keeping complex concepts which has all decomposed atomic concepts in the old ontology - but this may prevent the new combinations of atomic concepts? e.g. [EX.](<609096000> [EX.](<42752001> <94602001>))-SCTID_NULL in Disease TODO
                            list_parents_new.remove(parent)
                            print('list_parents_new:',list_parents_new, 'parent', parent, 'removed')
                            break  
        return list_parents_new
    #print('start')
    list_parents_new_final = get_in_KB_direct_parents_recur(onto,concept_iri,dict_SCTID_onto,dict_SCTID_onto_filtering,onto_verbaliser,concept_type=concept_type)
    #print('list_parents_new_final:',list_parents_new_final)
    #print('finish')
    return list_parents_new_final

# def get_in_KB_siblings(onto,concept_iri,dict_SCTID_onto,dict_SCTID_onto_filtering):
#     '''return a direct list of parent or a direct set of children ids, which are in the old ontology'''
#     list_parents = get_in_KB_direct_parents(onto,concept_iri,dict_SCTID_onto,dict_SCTID_onto_filtering)

def get_entity_graph_info(onto,concept_iri,dict_SCTID_onto,dict_SCTID_onto_filtering=None,onto_older=None,onto_verbaliser=None,concept_type="atomic",allow_complex_edge=False):
    '''
    get the direct parents, children, direct parent-entity-chidren paths (each as a tuple), and siblings of the entity - all atomic as default

    concept_type: "atomic", "complex", "all" (or any other strings)

    allow_complex_edge, used when concept_type is "all", set as True to allow complex edges
    '''
    # combine the results of "atomic" and "complex" if reporting "all", to note that edges only contain atomic concepts
    if concept_type == "all":        
        pc_info_str_tuple_atomic = get_entity_graph_info(onto,concept_iri,dict_SCTID_onto,dict_SCTID_onto_filtering=dict_SCTID_onto_filtering,onto_older=onto_older,onto_verbaliser=onto_verbaliser,concept_type="atomic")
        pc_info_str_list_atomic = list(pc_info_str_tuple_atomic)

        pc_info_str_tuple_complex = get_entity_graph_info(onto,concept_iri,dict_SCTID_onto,dict_SCTID_onto_filtering=dict_SCTID_onto_filtering,onto_older=onto_older,onto_verbaliser=onto_verbaliser,concept_type="complex")
        pc_info_str_list_complex = list(pc_info_str_tuple_complex)

        children_str_atomic, parents_str_atomic, _, _, _  = pc_info_str_list_atomic
        children_str_complex, parents_str_complex, _, _, _ = pc_info_str_list_complex
    
        # if parents_str_complex != '':
        #     print('parents_str_complex:', parents_str_complex)
            
        if allow_complex_edge:
            # update pc_paths_str_with_complex with complex concepts
            list_paths_with_complex = []
            list_children_atomic = children_str_atomic.split("|") if children_str_atomic != "" else []
            list_parents_atomic = parents_str_atomic.split("|") if parents_str_atomic != "" else []
            list_children_complex = children_str_complex.split("|") if children_str_complex != "" else []
            list_parents_complex = parents_str_complex.split("|") if parents_str_complex != "" else []

            for child_complex in list_children_complex:            
                for parent_atomic in list_parents_atomic:
                    list_paths_with_complex.append((parent_atomic,child_complex))
                # add THING-complexConcept if no atomic parent    
                if len(list_parents_atomic) == 0:
                    list_paths_with_complex.append((CONST_THING_NODE,child_complex))        
            for parent_complex in list_parents_complex:            
                for child_atomic in list_children_atomic:
                    list_paths_with_complex.append((parent_complex,child_atomic))
                # add complexConcept-NULL if no atomic child
                if len(list_children_atomic) == 0:
                    list_paths_with_complex.append((parent_complex,CONST_NULL_NODE))    
            for parent_complex in list_parents_complex:
                for child_complex in list_children_complex:
                    list_paths_with_complex.append((parent_complex,child_complex))
            # if len(list_paths_with_complex) > 0:
            #     print('list_paths_with_complex (non-empty):',list_paths_with_complex)        
            pc_paths_str_with_complex = '|'.join([pc_tuple_with_complex[0] + '-' + pc_tuple_with_complex[1] for pc_tuple_with_complex in list_paths_with_complex])
            pc_info_str_list_complex[2] = pc_paths_str_with_complex

        # merge all output elements in atomic and complex modes
        pc_info_str_list = []
        for pc_info_str_atomic, pc_info_str_complex in zip(pc_info_str_list_atomic,pc_info_str_list_complex):
            #print('pc_info_str_atomic:',pc_info_str_atomic,'pc_info_str_complex:',pc_info_str_complex)
            pc_info_str = pc_info_str_atomic + "|" + pc_info_str_complex if pc_info_str_complex != "" else pc_info_str_atomic
            pc_info_str_list.append(pc_info_str)
        #print('len(pc_info_str_list):',len(pc_info_str_list))
        # for concept_type_ in ["atomic", "complex"]:
        #     pc_info_str_tuple_ = get_entity_graph_info(onto,concept_iri,dict_SCTID_onto,dict_SCTID_onto_filtering=dict_SCTID_onto_filtering,onto_older=onto_older,onto_verbaliser=onto_verbaliser,concept_type=concept_type_)
            
        #     # update pc_path if there are complex concepts as parents (and/or children)
        #     if concept_type_ == "complex":

        #     if concept_type_ == "complex":
        #         children_str_complex, parents_str_complex, pc_paths_str_complex, children_tit_str_complex, parents_tit_str_complex = pc_info_str_tuple_

        #         pass

        #     for ind,pc_info_str_ in enumerate(pc_info_str_tuple_):
        #         if pc_info_str_ != '':
        #             if pc_info_str_list[ind] != '':
        #                 pc_info_str_list[ind] = pc_info_str_list[ind] + '|' + pc_info_str_
        #             else:
        #                 pc_info_str_list[ind] = pc_info_str_

        return tuple(pc_info_str_list)

    list_parents = get_in_KB_direct_parents(onto,concept_iri,dict_SCTID_onto,dict_SCTID_onto_filtering,onto_verbaliser,concept_type=concept_type)
    list_children = get_in_KB_direct_children(onto,concept_iri,dict_SCTID_onto,dict_SCTID_onto_filtering,onto_verbaliser,concept_type=concept_type)

    if concept_type == "complex":
        list_parents_rangeNodes = list_parents[:]
        list_children_rangeNodes = list_children[:]
        #print("list_parents_rangeNodes:",list_parents_rangeNodes)
        list_parents = [clean_id_in_parsed_owl_class_expression(parent_rangeNode.text) for parent_rangeNode in list_parents_rangeNodes if not filter_complex_concept(parent_rangeNode.text)] # also filter the list of complex parents
        list_children = [clean_id_in_parsed_owl_class_expression(child_rangeNode.text) for child_rangeNode in list_children_rangeNodes if not filter_complex_concept(child_rangeNode.text)] # also filter the list of complex children        
        # filter the list of complex parents and children
        # list_parents = [complex_parent for complex_parent in list_parents if not filter_complex_concept(complex_parent)]
        # list_children = [complex_children for complex_children in list_children if not filter_complex_concept(complex_children)]

    parents_str = '|'.join(list_parents)
    children_str = '|'.join(list_children)
    
    # form parent-child paths only when the both concepts are atomic
    if concept_type == "atomic":
        list_paths = []
        if len(list_parents) == 0:
            for child in list_children:
                list_paths.append((CONST_THING_NODE,child))
        if len(list_children) == 0:
            for parent in list_parents:
                list_paths.append((parent,CONST_NULL_NODE))

        for child in list_children:
            for parent in list_parents:
                list_paths.append((parent,child))

        pc_paths_str = '|'.join([pc_tuple[0] + '-' + pc_tuple[1] for pc_tuple in list_paths])
    else:
        pc_paths_str = ''

    #get the children and parent titles
    if concept_type == "atomic":
        # get the children and parent titles (for atomic concepts) - from the older onto as they should all be in the older onto
        onto_older = onto_older if onto_older != None else onto # the older version is used for filtering (when newer ontology is used as the main) or as the main ontology
        
        children_tit_str = get_concept_tit_from_id_strs(onto_older,children_str)
        parents_tit_str = get_concept_tit_from_id_strs(onto_older,parents_str)
    if concept_type == "complex":
        #print("list_parents_rangeNodes:",list_parents_rangeNodes)
        children_tit_str = '|'.join([verbalise_concept_CfgNode(verbalise_concept(onto_verbaliser, child_rangeNode)) for child_rangeNode in list_children_rangeNodes])
        parents_tit_str = '|'.join([verbalise_concept_CfgNode(verbalise_concept(onto_verbaliser, parent_rangeNode)) for parent_rangeNode in list_parents_rangeNodes])

    return children_str, parents_str, pc_paths_str, children_tit_str, parents_tit_str

def get_concept_tit_from_id_strs(onto,id_strs):
    if id_strs == '':
        return ''

    list_ids = id_strs.split('|')
    list_id_tits = [get_title_in_onto_from_iri(onto,get_iri_from_SCTID_id(id)) for id in list_ids]
    return '|'.join(list_id_tits)

def filter_complex_concept(complex_concept_id):
    '''
        return True if the complex concept is to be filtered: i.e. 
        (i) can be decomposed by conjunction (starting with [AND])
        (ii) do not contain [EX.]
    '''
    return (not "[EX.]" in complex_concept_id) or complex_concept_id.startswith("[AND]")