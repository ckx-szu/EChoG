# EChoG

A synchronized ECG–Echo dataset and benchmark for reference-frame-guided echocardiographic video generation.

## Overview

EChoG combines A4C Echo videos, synchronized ECG traces, and patient-level diagnosis-derived labels. The benchmark studies video reconstruction quality, reliance on ECG timing, and preservation of disease-related information.

The cohort contains 3,607 paired recordings from 1,437 patient identity groups. At least one disease label is valid for 925 patient groups, corresponding to 2,245 recordings. Missing or uncertain labels are masked rather than treated as negative.

## Generation task

The generator receives a reference Echo image and a target ECG window whose starting cardiac phase need not match the reference image. Reference and target data are drawn from the same source recording. Explicit phase annotations and disease labels are not inference inputs.

These clinical recordings are not wearable ECG recordings. Generalization across examinations and to everyday wearable monitoring remains to be validated. ECG traces are extracted from the ultrasound display rather than provided as calibrated native device waveforms.

## Disease categories

| Code | Category |
| --- | --- |
| CM | Cardiomyopathy |
| HF | Heart failure / cardiac dysfunction |
| MV | Mitral valve disease |
| TV | Tricuspid valve disease |
| PH | Pulmonary hypertension |
| SD | Atrial / ventricular septal defects |

These are clinical-text-derived research labels, not a claim that a single A4C video establishes all six diagnoses. Categories may overlap.

## Current repository status

This is a repository scaffold. Patient data, record-level manifests, executable benchmark code, trained weights, and dataset download links are not included yet. No runnable benchmark or public dataset release is claimed by this scaffold.

## Planned release contents

- Data loading and reference-frame review tools
- Patient-level splits and de-identified record indexes
- Baseline configurations and evaluation scripts
- Data access instructions and documented label semantics

## Data access and terms

Data access instructions will be added after release review. Do not upload identifiable DICOM files, clinical spreadsheets, original identity mappings, credentials, or unreviewed clinical media. Code and data licensing terms will be specified separately before release.

