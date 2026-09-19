# Turkish NLP Projects

Jupyter notebooks exploring Turkish text classification with both classical machine learning and transformer models.

## Notebooks

| Notebook | Task | Approach |
| :--- | :--- | :--- |
| `turkish_sentiment_berturk_xlmr.ipynb` | Turkish sentiment analysis | Fine-tuning **BERTurk** (`dbmdz/bert-base-turkish-cased`) and **XLM-RoBERTa** (`xlm-roberta-base`), compared on the same split |
| `turkish_sms_spam_classical_ml.ipynb` | Turkish SMS spam detection | Text cleaning, TF-IDF features and classical classifiers (Logistic Regression, Linear SVM, Naive Bayes) |
| `case1.ipynb` | Case study | End-to-end exploratory NLP pipeline |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn jupyter
jupyter notebook
```

> Transformer fine-tuning is GPU-friendly; Google Colab works well for the BERTurk / XLM-R notebook.

## Related work

- [Sina — Turkish Health Assistant](https://github.com/semihbekdas/yapay-zeka-saglik-asistani-sina): 16-class medical specialty routing with BERTurk / XLM-R plus a LoRA-tuned Llama 3.1 chat assistant.

## Author

**Semih Bekdaş** · [GitHub](https://github.com/semihbekdas) · [Hugging Face](https://huggingface.co/SemihBekdas)
