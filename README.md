# Clinical Entity Linking System for Greek hospital discharge documents with ICD 10 codes

This project focuses on implementing an entity-linking system tailored for Greek medical documents. The system features a hierarchical classifier designed to handle entity-linking tasks within pre-extracted mentions, capitalizing on the inherent hierarchical structure of the ICD-10 coding system. Complementing the classifier is a bi-encoder, which introduces a candidate generation step. 
The system is additionally evaluated under the scenario of known document-level labels. 

## Contents:

1. Class files 
   - HierarchyClassifier.py: contains the class that implements the Hierarchical Classifier
   - ICD10Encoder.py: contains the class that implements the ICD 10 Bi-encoder

2. Training and Evaluation files
   - Hierarchical_training.ipynb: sample training and evaluation of the Hierarchy Classifier
   - BiEncoder_training.ipynb: sample training of the Bi-encoder
   - biencoderCG.ipynb: evaluation of the system as a two-staged approach containing a candidate generation step using the bi encoder and a reranking step using the hierarchical classifier
   - document_level.ipynb: evaluation of the classifier and the bi-encoder given document-level labels
