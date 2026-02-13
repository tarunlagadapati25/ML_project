# ML_project

# Census Income Machine Learning Project

## 1. Project Objective

This project aims to:

1. Build a **classification model** to predict whether a person earns:
   - `<= $50,000`
   - `> $50,000`
2. Develop a **customer segmentation model** to group individuals for marketing purposes.

The dataset consists of **40 demographic and employment-related features** extracted from **U.S. Census data**.

---

## 2. Project Structure

census-income-ml-project/<br>
│<br>
├── data/<br>
│ ├── census-bureau.data<br>
│ └── census-bureau.columns<br>
│<br>
├── src/<br>
│ ├── data_preprocessing.py<br>
│ ├── classification.py<br>
│ ├── segmentation.py<br>
│<br>
├── requirements.txt<br>
├── README.md<br>


---

## 3. Installation Instructions

### 3.1 Clone the repository

Copy and paste in your terminal:

```bash
git clone https://github.com/tarunlagadapati25/ML_project.git
cd ML_project
```
### 3.2 Create a virtual environment

Mac/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
### 3.3 Install dependencies
```bash
pip install -r requirements.txt
```

# 4. Running the Models
### 4.1 Classification Model
```bash
cd src
python classification.py
```


Expected Output:

Accuracy score
Precision, Recall, F1-score



### 4.2 Segmentation Model
```bash
cd src
python segmentation.py
```

Expected Output:
`Cluster distribution information`

# 6. Business Application

### Classification model helps:

`Identify high-income individuals for premium product targeting.`

### Segmentation model helps:

Personalize marketing campaigns<br>
Identify demographic clusters<br>
Allocate advertising budget efficiently

# 7. Requirements

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```
# 8. Author

Tarun Lagadapati.