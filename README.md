# Greek Hospital Entity Linking System with ICD-10 Codes

This project implements a **Clinical Entity Linking System** designed for Greek hospital discharge documents. The system is tailored to handle medical entities and their relations within these documents, particularly focusing on linking entities to the **ICD-10 codes**. The approach uses both a **hierarchical classifier** and a **bi-encoder** to process and link extracted medical mentions.

The project is based on my **Master's Thesis**, which explains the theory, methodology, and detailed implementation of the entity linking system.

## Master's Thesis

The full master's thesis, titled **"Clinical Entity Linking for Greek Hospital Discharge Documents with ICD-10 Codes"**, can be found in the [docs](docs) folder of this repository:

- [Download Master's Thesis (PDF)](docs/masters_thesis.pdf)

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
- **document_level.ipynb**: Evaluates the system with document-level labels to assess the full entity linking pipeline.

## Setup and Installation

To get started with the project, clone the repository and install the required dependencies.

### Clone the repository:
```bash
git clone https://github.com/your-username/Greek-Hospital-Entity-Linking-ICD10.git
cd Greek-Hospital-Entity-Linking-ICD10





~~~~~~~~~~~~~~~~~~~



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
