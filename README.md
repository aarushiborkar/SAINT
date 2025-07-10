# SAINT

SAINT is a multilingual project designed for sentiment classification and explainability in both high-resource and low-resource languages.

It currently supports:

- **German**
- **Amharic**

with functionality for both:

- **Sentence-Level Classification**
- **Span/Token-Level Classification**

# 📁 Project Structure

```
saint/
├── german_sentiment/
│   ├── prepare_data.py
│   ├── train_transformer.py
│   ├── explain_captum.py
│   ├── data_processing.py
│   ├── span_level.py
│   ├── bert.py
│   ├── gpt.py
│   ├── flan.py
│   ├── captum_explainability.py
│   ├── generate_comparison_html.py
│   ├── models/
│   └── outputs/
│
├── amharic_sentiment/
│   └── (mirrors the german_sentiment structure)
```

# ✅ Features

## Multilingual Support

- German
- Amharic

## Classification Levels

- Sentence-level classification
- Span-level / token-level classification with BIO tagging

## Transformer Models Used

- BERT (`bert-base-german-cased`)
- GPT-2
- FLAN-T5

## Explainability

- Captum-based explainability for both sentence and token levels

## Error Analysis

- Outputs merged into a visual HTML report for cross-model comparison

# 🔧 Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/saint.git
cd saint
```

## Set Up a Python Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

# 🚀 Usage

Choose your working directory:  
`german_sentiment/` or `amharic_sentiment/`

## Sentence-Level Classification

### Step 1: Prepare the Dataset

```bash
python prepare_data.py
```

### Step 2: Train the Transformer Model

```bash
python train_transformer.py
```

### Step 3: Generate Explainability Outputs (Captum)

```bash
python explain_captum.py
```

#### Output Files

- `outputs/test_predictions.csv`
- `outputs/test_token_attributions.csv`

## Span-Level / Token-Level Classification

### Step 1: Convert and Split Data into BIO Format

```bash
python data_processing.py
```

### Step 2: Perform Span-Level Predictions

```bash
python span_level.py
```

### Step 3: Train on BIO-Tagged Token Data

```bash
python bert.py
```

### Step 4: Generate Token-Level Explainability (Captum)

```bash
python captum_explainability.py
```

#### Output Files

- `outputs/token_predictions.csv`
- `outputs/token_attributions.csv`

## GPT-2 Based Classification

### Step 1: Run GPT-2 Inference

```bash
python gpt.py
```

#### Output

- `outputs/test_predictions_gpt2.csv`

## FLAN-T5 Based Classification

### Step 1: Run FLAN-T5 Prompt-based Classification

```bash
python flan.py
```

#### Output

- `outputs/test_predictions_flan.csv`

## HTML Comparison Report

### Step 1: Generate Comparison HTML for Error Analysis

```bash
python generate_comparison_html.py
```

#### Output

- `comparison_file.html`

This file visually compares model predictions from BERT, GPT-2, and FLAN side-by-side.

# 📄 Output Files Summary

| Filename                      | Description                                    |
|------------------------------|------------------------------------------------|
| test_predictions.csv         | Sentence-level BERT predictions                |
| test_token_attributions.csv  | Captum sentence-level attributions             |
| token_predictions.csv        | Token-level span predictions (BIO format)      |
| token_attributions.csv       | Captum token-level attributions                |
| test_predictions_gpt2.csv    | GPT-2 model predictions                        |
| test_predictions_flan.csv    | FLAN-T5 model predictions                      |
| comparison_file.html         | HTML report for visual comparison              |

# 🧠 Models Used

- `bert-base-german-cased`
- `gpt2`
- `google/flan-t5-base`

# 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

# 👩‍💻 Author

**Aarushi Borkar**

(Add your GitHub, LinkedIn, or email here if you'd like)
