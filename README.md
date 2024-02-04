# Clinical Entity Linking System for Greek hospital discharge documents with ICD 10 codes

## Contents:

1. .py files
   - HierarchyClassifier.py: contains the class that implements the Hierarchical Classifier
   - ICD10Encoder.py: contains the class that implements the ICD 10 Bi-encoder

2. .ipynb files
   - Hierarchical_training.ipynb: sample training of the Hierarchy Classifier
   - BiEncoder_training.ipynb: sample training of the Bi-encoder
   - biencoderCG.ipynb: evaluation of the system as a two-staged approach containing a candidate generation step using the bi encoder and a reranking step using the hierarchical classifier
   - document_level.ipynb: evaluation of the classifier and the bi-encoder given document-level labels
