# Real vs Fake Job Posting Classifier

Detect fraudulent job listings using NLP and machine learning.  
Models include Random Forest + SMOTE and Neural Network.  
Built using Python, pandas, scikit-learn, spaCy, and TF-IDF vectorization.

## Dataset

Original: [Kaggle - Real or Fake Job Posting](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)  
Size: ~17,880 rows × 18 columns  
Note: Not uploaded due to GitHub file size limit. See `/data/README.md` for setup.

## Project Structure

```
fake-job-posting-detector/
├── data/
│   └── README.md                  # Instructions to download large CSV
│
├── notebooks/
│   ├── Main.ipynb                 # EDA + preprocessing
│   ├── RFC_SMOTE.ipynb            # Random Forest + SMOTE model
│   └── Neural_Network.ipynb       # Neural Net with SMOTE model
│
├── .gitignore                     # Ignores large CSVs in /data
├── README.md                      # Full project documentation
├── requirements.txt               # Package list to reproduce (optional)
```


## Approach

- **Text Cleaning**: Lowercase, punctuation, stopwords removal
- **NLP**: Lemmatization, POS tagging (spaCy)
- **Vectorization**: TF-IDF
- **Class Imbalance Handling**: SMOTE
- **Models**:
  - Random Forest (with SMOTE)
  - Neural Network (with SMOTE)
