# Clinical Entity Linking System for Greek hospital discharge documents with ICD 10 codes

This project focuses on implementing an entity-linking system tailored for Greek medical documents. The system features a hierarchical classifier designed to handle entity-linking tasks within pre-extracted mentions, capitalizing on the inherent hierarchical structure of the ICD-10 coding system. Complementing the classifier is a bi-encoder, which introduces a candidate generation step. 
The system is additionally evaluated under the scenario of known document-level labels. 

The project is part of my **Master's Thesis**, which explains the theory, methodology, and detailed implementation of the entity linking system.

## Overview

The system utilizes the hierarchical structure of ICD-10 codes for entity linking and builds a two-step process:
1. **Hierarchical Classifier**: Handles entity linking by leveraging the hierarchical relationships in ICD-10 codes.
2. **Bi-Encoder**: Provides a candidate generation step that selects relevant ICD-10 codes for each mention, which is then reranked by the hierarchical classifier.

Additionally, the system is evaluated under both **document-level labels** and **mention-level entity linking** scenarios.

## Key Components

### Class Files
- **HierarchyClassifier.py**: Implements the Hierarchical Classifier to predict ICD-10 code associations for medical mentions.
- **ICD10Encoder.py**: Implements the Bi-Encoder model for candidate generation.

### Training and Evaluation Notebooks
- **Hierarchical_training.ipynb**: Contains sample training and evaluation of the Hierarchical Classifier.
- **BiEncoder_training.ipynb**: Demonstrates training of the Bi-Encoder model.
- **biencoderCG.ipynb**: Evaluates the two-step approach, combining candidate generation with the hierarchical classifier reranking.
- **document_level.ipynb**: Evaluates the of the classifier and the bi-encoder given document-level labels.
