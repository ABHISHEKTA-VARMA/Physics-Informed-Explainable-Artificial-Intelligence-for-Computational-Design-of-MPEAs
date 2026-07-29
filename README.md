# Physics-Informed Explainable Artificial Intelligence for Computational Design of Multi-Principal Element Alloys

## Overview

This repository contains the computational workflow developed for the physics-informed and explainable artificial intelligence framework presented in the accompanying research work. The workflow integrates physics-based descriptor engineering, machine learning, explainable artificial intelligence (XAI), finite element analysis (FEA), multi-criteria decision-making (MCDM), and density functional theory (DFT) input generation for the computational design and assessment of multi-principal element alloys (MPEAs).

The repository has been organized to follow the complete computational methodology used in the study, beginning with descriptor generation from alloy compositions and ending with the preparation of first-principles simulation inputs for selected candidate alloys.

---

# Repository Workflow

```
Raw Alloy Dataset
        │
        ▼
Miedema Database Construction
        │
        ▼
Dataset Preprocessing
        │
        ▼
Physics-Informed Descriptor Engineering
        │
        ▼
Machine Learning Development
        │
        ▼
Model Evaluation
        │
        ▼
Applicability Domain Analysis
        │
        ▼
Explainable AI Analysis
        │
        ▼
Candidate Alloy Generation
        │
        ▼
Validation Dataset Preparation
        │
        ▼
Finite Element Analysis Workflow
        │
        ▼
Multi-Criteria Decision Making
        │
        ▼
DFT Input Preparation
```

---

# Repository Structure

```
src/

1.1_Generate_Miedema_Enthalpy_Template.py
1.2_Populate_Miedema_Enthalpy_Database.py
1.3_Verify_Miedema_Enthalpy_Database.py

2_Dataset_Preprocessing.py

3.1_Calculate_Physics_Informed_Descriptors.py
3.2_Dataset_Aggregation_and_Quality_Control.py
3.3_Finalize_Machine_Learning_Dataset.py

4.1_Model_Development_and_Optimization.py
4.2_Model_Evaluation_and_Visualization.py
4.3_Build_Applicability_Domain.py
4.4_Generate_Descriptors_for_Candidate_Alloys.py
4.5_Explainable_AI_and_Descriptor_Interpretation.py

5_Generative_Alloy_Design_and_Ranking.py

6_Stratified_Validation_Dataset_Preparation.py

7.1_Generate_FEM_Material_Cards.py
7.2_Mesh_Convergence_Analysis.py
7.3_ML_FEM_Comparative_Assessment.py

8_MCDM_Ranking_and_Sensitivity_Analysis.py

9_Generate_DFT_Input_Files.py
```

---

# Computational Workflow

### Step 1 – Binary Mixing Enthalpy Database

Constructs and verifies the binary interaction database required for thermodynamic descriptor calculation.

### Step 2 – Dataset Preprocessing

Processes the experimental alloy database through composition parsing, normalization, metadata standardization, duplicate handling, and quality control.

### Step 3 – Physics-Informed Descriptor Engineering

Calculates compositional, thermodynamic, structural, and derived descriptors used as machine learning inputs.

### Step 4 – Machine Learning and Explainable AI

Develops predictive machine learning models, evaluates predictive performance, defines the applicability domain, generates descriptors for candidate alloys, and interprets model behaviour using explainable artificial intelligence methods.

### Step 5 – Candidate Alloy Generation

Generates candidate alloy compositions, predicts hardness, evaluates novelty and applicability domain, and ranks candidate alloys using multiple screening criteria.

### Step 6 – Validation Dataset Preparation

Constructs stratified validation datasets for subsequent finite element analysis and computational assessment.

### Step 7 – Finite Element Analysis Preparation and Assessment

Generates constitutive material cards for finite element simulations, evaluates mesh convergence, and compares machine learning predictions with finite element estimates.

### Step 8 – Multi-Criteria Decision Making

Ranks candidate alloys using objective weighting and TOPSIS-based multi-criteria decision-making together with sensitivity analysis.

### Step 9 – Density Functional Theory Input Preparation

Generates Quantum ESPRESSO input files for selected candidate alloys to facilitate first-principles calculations.

---

# Software Requirements

The workflow has been developed using Python 3.

Major Python packages include:

- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib
- Seaborn
- SHAP
- Joblib
- Statsmodels
- XGBoost
- LightGBM
- CatBoost
- Matminer
- pymatgen

Additional software used during the computational workflow includes:

- ANSYS Mechanical
- Quantum ESPRESSO

---

# Installation

Clone the repository

```bash
git clone https://github.com/<username>/<repository>.git
```

Move into the project directory

```bash
cd <repository>
```

Install the required Python packages

```bash
pip install -r requirements.txt
```

---

# Input Data

The workflow requires:

- Experimental MPEA dataset
- Binary mixing enthalpy database
- Elemental property database

The required input files should be placed in the directories expected by each script.

---

# Outputs

The workflow generates:

- Processed datasets
- Physics-informed descriptors
- Trained machine learning models
- Explainable AI analyses
- Candidate alloy rankings
- Applicability domain models
- Finite element material cards
- Mesh convergence analyses
- ML–FEM comparative assessment
- Multi-criteria ranking results
- Quantum ESPRESSO input files

---

# Citation

If this repository contributes to your research, please cite the accompanying publication once available.

---

# License

License information will be provided with the final public release of the repository.

---

# Contact

For questions regarding the computational workflow, please contact the corresponding author of the accompanying publication.
