# Code for MSE-ENN

Official code repository for the paper **"Zero-shot New Drug Interaction Prediction via Multimodal Synergy-Enhanced Representation Learning."**

## Overview

MSE-ENN is designed for **zero-shot drug–drug interaction (DDI) prediction**, where new drugs have no known DDI labels during training. The model integrates complementary drug information to learn representations that generalize to interactions involving unseen drugs.

The framework considers multiple sources of drug information, including drug–protein associations, molecular substructures, SMILES-based representations, and therapeutic descriptions. It also includes analyses to examine the contributions of different modalities and features to DDI predictions.

## Repository Structure

| Directory | Description |
| --- | --- |
| `Dataset1/` | Data for Dataset 1. |
| `Dataset2/` | Data for Dataset 2. |
| `Dataset3/` | Data for Dataset 3. |
| `MSE-ENN_Dataset1/` | MSE-ENN implementation and experiments for Dataset 1. |
| `MSE-ENN-Dataset2/` | MSE-ENN implementation and experiments for Dataset 2. |
| `MSE-ENN_Dataset3/` | MSE-ENN implementation and experiments for Dataset 3. |
| `MSE-ENN-RDFP/` | **Embedding-level interpretation** using RDFP to analyze the sensitivity or contribution of learned drug representations to DDI predictions. |
| `MSE-ENN_explain/` | **PairSHAP-based interpretation** to quantify interaction contributions between pairs of modalities or features in DDI predictions. |
