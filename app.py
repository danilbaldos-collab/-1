import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# Настройка страницы Streamlit
st.set_page_config(page_title="Анализ тональности рецензий", layout="wide")

st.title("🎬 Проект: Анализ тональности рецензий на фильмы")
st.markdown("**Стек технологий:** Python, Streamlit, NLTK, Scikit-learn, Matplotlib/Seaborn")


# Кэшируем тяжелые операции, чтобы сайт не переобучал модель при каждом клике
@st.cache_resource
def load_and_train_model():
    nltk.download('movie_reviews', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    from nltk.corpus import movie_reviews, stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    texts, labels = [], []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            texts.append(movie_reviews.raw(fileid))
            labels.append(1 if category == 'pos' else 0)

    df = pd.DataFrame({'review': texts, 'sentiment': labels})

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        tokens = word_tokenize(text)
        return ' '.join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words])

    df['cleaned_review'] = df['review'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2,
                                                        random_state=42)

    vectorizer = CountVectorizer(max_features=3000, binary=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    return model, vectorizer, X_test, X_test_vec, y_test, preprocess_text


# Запуск обучения (происходит один раз при старте)
with st.spinner("Загрузка датасета и обучение модели логистической регрессии..."):
    model, vectorizer, X_test, X_test_vec, y_test, preprocess_fn = load_and_train_model()

# Метрики
y_pred = model.predict(X_test_vec)
y_proba = model.predict_proba(X_test_vec)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# --- ИНТЕРФЕЙС САЙТА ---

# Блок 1: Интерактивная демо-форма
st.header("🔍 Демо-форма для проверки своей рецензии")
user_input = st.text_area("Введите текст отзыва на английском языке:",
                          placeholder="This movie was absolutely brilliant! Great acting and execution...")

if st.button("Проверить тональность"):
    if user_input.strip():
        cleaned = preprocess_fn(user_input)
        vec = vectorizer.transform([cleaned])
        prob = model.predict_proba(vec)[0][1]

        if prob > 0.5:
            st.success(f"🍿 **Позитивный отзыв** (Уверенность модели: {prob * 100:.1f}%)")
        else:
            st.error(f"😡 **Негативный отзыв** (Уверенность модели: {(100 - prob * 100):.1f}%)")
    else:
        st.warning("Пожалуйста, введите текст для анализа.")

st.divider()

# Блок 2: Графики метрик
st.header("📊 Метрики качества модели")
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Confusion Matrix (Accuracy: {accuracy:.4f})")
    fig_cm, ax_cm = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    st.pyplot(fig_cm)

with col2:
    st.subheader(f"ROC Curve (ROC-AUC: {roc_auc:.4f})")
    fig_roc, ax_roc = plt.subplots()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend()
    st.pyplot(fig_roc)

st.divider()

# Блок 3: Важность слов
st.header("💡 Анализ важности слов (Какие слова влияют на вердикт)")
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]
word_importance = pd.DataFrame({'Слово': feature_names, 'Вес (Важность)': coefs})

col_pos, col_neg = st.columns(2)
with col_pos:
    st.subheader("🔝 ТОП-10 маркеров ПОЗИТИВА")
    st.dataframe(word_importance.sort_values(by='Вес (Важность)', ascending=False).head(10), use_container_width=True)

with col_neg:
    st.subheader("🛑 ТОП-10 маркеров НЕГАТИВА")
    st.dataframe(word_importance.sort_values(by='Вес (Важность)', ascending=True).head(10), use_container_width=True)