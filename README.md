# Email Spam Detector

A reproducible text spam classification demo using CountVectorizer and Multinomial
Naive Bayes. Despite the project name, the bundled examples are SMS messages:
results do not establish performance on real email, sender metadata, or attachments.

## Setup

Use Python 3.12 or newer. From the project directory:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell with `.venv\Scripts\Activate.ps1`,
or on macOS/Linux with `source .venv/bin/activate`. Then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train, evaluate, and predict

```bash
python spam_detector.py
python spam_detector.py --message "Congratulations! Claim your free prize now"
python spam_detector.py --data path/to/messages.csv
```

The command prints held-out accuracy, per-class precision/recall/F1, and a
confusion matrix. Matrix rows are actual labels and columns are predicted labels,
in the printed `label_order`. With `--message`, it then refits on the full cleaned
dataset and prints `ham` (legitimate) or `spam`.

For plots and exploration:

```bash
jupyter notebook email_spam_detector.ipynb
```

Start Jupyter from this project directory and run all notebook cells in order.

## Data and evaluation

`spam.csv` contains 5,572 labeled messages with `Category` (`ham` or `spam`) and
`Message` columns. Missing values, blank messages, unknown labels, and conflicting
labels are rejected. Repeated messages are removed before an 80/20 stratified
split with random seed 42. The vectorizer learns only from the training split.
Accuracy alone can hide missed spam, so inspect spam precision and recall too.

The included data arrived with this project; its original attribution and
redistribution terms are not documented here. No new license is asserted for it.
Use the model as a learning demo and validate on representative data before
using predictions to filter messages. Do not commit private messages or credentials.

## Checks

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs tests and executes the notebook. Dependencies have compatible
version ranges; exact results can vary between dependency versions.

`docs/engineering_log.md` is a pre-existing historical document. Its experimental
claims have not been verified against this implementation.
