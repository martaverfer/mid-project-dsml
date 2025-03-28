# Installation

To run this project locally, follow these instructions:

## 1. **Clone this repository:**

```bash
git clone https://github.com/martaverfer/mid-project-dsml.git \
cd notebooks
```

## 2. **Virtual environment:**

Create the virtual environment: 
```bash
python3 -m venv venv
```

Activate the virtual environment:

- For Windows: 
```bash
venv\Scripts\activate
```

- For Linux/Mac: 
```bash
source venv/bin/activate
```

To switch back to normal terminal usage or activate another virtual environment to work with another project run:
```deactivate```

## 3. **Install dependencies:**

```bash
pip install --upgrade pip; \
pip install -r requirements.txt
```

## 4. **Open the Jupyter notebook to explore the analysis:**

```bash
cd nootebooks; \
eda.ipynb
improve_models.ipynb
model_evaluation.ipynb
```

This script will execute the analysis steps and produce the results.

## 5. **For Running Streamlit App**
```bash
cd app; \
streamlit run app.py
```