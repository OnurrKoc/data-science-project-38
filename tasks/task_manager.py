"""
DS-38 — Film Yorumu Duygu Analizi (TF-IDF + Logistic Regression)
Bir streaming / e-ticaret şirketinde data scientist'sin. Kullanıcı yorumlarını
otomatik olarak pozitif (pos) / negatif (neg) etiketleyen bir duygu analizi
(sentiment analysis) modeli kuruyorsun. TF-IDF + Logistic Regression — NLP'nin
klasik ve hâlâ çok güçlü baseline'ı.

Her fonksiyonun pass kısmını doldur. Testleri çalıştır, hepsi geçene kadar
iterate et: `python watch.py` veya `pytest tests/test_question.py -v`
"""
import nltk
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

# 1. NLTK movie_reviews corpus'unu indir ve (text, label) listesi döndür
def fetch_reviews():
    """
    NLTK movie_reviews corpus'unu indir ve her yorum için (text, label)
    ikilisinden oluşan bir liste döndür.

    Akış:
    1. nltk.download('movie_reviews', quiet=True)  → ilk seferde indirir,
       sonra ~/nltk_data cache'inden okur (URL/SSL ile uğraşmana gerek yok)
    2. from nltk.corpus import movie_reviews
    3. movie_reviews.fileids() → 2000 dosya id'si (1000 pos + 1000 neg)
    4. Her fileid için:
       - label = movie_reviews.categories(fid)[0]   # 'pos' veya 'neg'
       - text  = movie_reviews.raw(fid)             # ham yorum metni
    5. (text, label) ikililerinden bir liste döndür (2000 eleman)

    Returns:
        list: [(text, label), ...] — 2000 eleman, label 'pos'/'neg'

    İpucu: import nltk; import içini fonksiyon içinde yapabilirsin.
    - nltk.download'ı movie_reviews import'undan ÖNCE çağır.
    - data = []; for fid in movie_reviews.fileids(): data.append((...))
    """
    nltk.download('movie_reviews', quiet=True)
    from nltk.corpus import movie_reviews
    data = []
    for fid in movie_reviews.fileids():
        label = movie_reviews.categories(fid)[0]   # 'pos' veya 'neg'
        text = movie_reviews.raw(fid)
        data.append((text, label))
    return data


# 2. (text, label) listesini DataFrame olarak yükle
def load_reviews(data=None):
    """
    Yorum verisini pandas DataFrame'e çevir.

    Args:
        data: fetch_reviews çıktısı [(text, label), ...]. None ise
              fonksiyon içinde fetch_reviews() çağrılır.

    Returns:
        pd.DataFrame: 2000 satır × 2 sütun ['text', 'label']

    İpucu:
    - if data is None: data = fetch_reviews()
    - pd.DataFrame(data, columns=['text', 'label'])
    """
    if data is None:
        data = fetch_reviews()
    return pd.DataFrame(data, columns=['text', 'label'])


# 3. Veriyi keşfet — label dağılımı
def explore_data(df):
    """
    Temel keşif metriği üret.

    Returns:
        dict: {
            'total': int (toplam yorum sayısı),
            'pos_count': int ('pos' = pozitif yorum sayısı),
            'neg_count': int ('neg' = negatif yorum sayısı),
            'balance': float (pos_count / total — veri dengeli, ~0.5)
        }

    İpucu:
    - df['label'].value_counts() → 'pos' ve 'neg' sayıları
    - total = len(df)
    """
    counts = df['label'].value_counts().to_dict()
    total = len(df)
    pos = int(counts.get('pos', 0))
    neg = int(counts.get('neg', 0))
    return {
        'total': total,
        'pos_count': pos,
        'neg_count': neg,
        'balance': pos / total,
    }


# 4. Label'ları encode et (pos/neg → 1/0)
def encode_labels(df):
    """
    'label' sütununu binary target'a çevir: 'pos' → 1, 'neg' → 0.
    Yeni bir 'target' sütununa yaz (orijinal 'label' kalsın).

    Args:
        df: 'label' sütunu olan DataFrame

    Returns:
        pd.DataFrame: yeni 'target' sütunu (int 0/1) eklenmiş kopya

    İpucu:
    - out = df.copy()
    - out['target'] = out['label'].map({'pos': 1, 'neg': 0}).astype(int)
    """
    out = df.copy()
    out['target'] = out['label'].map({'pos': 1, 'neg': 0}).astype(int)
    return out


# 5. Train/test split (stratified)
def split_data(X, y):
    """
    train_test_split kullan:
    - X: yorum metinleri (pd.Series — ham string'ler, vektörize EDİLMEMİŞ)
    - y: target (0/1)
    - test_size=0.2
    - stratify=y (pos/neg oranı train ve test'te korunsun)
    - random_state=42 (tekrarlanabilirlik)

    Returns:
        tuple: (X_train, X_test, y_train, y_test)

    İpucu: from sklearn.model_selection import train_test_split
    """
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# 6. TfidfVectorizer + LogisticRegression pipeline kur
def build_pipeline():
    """
    sklearn Pipeline:
    - 'tfidf': TfidfVectorizer(stop_words='english')
               → metni TF-IDF vektörüne çevirir (kelime sıklığı + ayırt
                 edicilik); stop_words='english' → 'the', 'a', 'is' gibi
                 anlamsız İngilizce kelimeleri eler
    - 'clf':   LogisticRegression(max_iter=1000)
               → TF-IDF vektörlerinden pozitif/negatif sınıfı öğrenir
                 (max_iter=1000 → yakınsama için yeterli iterasyon)

    Returns:
        sklearn.pipeline.Pipeline

    İpucu:
    - from sklearn.feature_extraction.text import TfidfVectorizer
    - from sklearn.linear_model import LogisticRegression
    - from sklearn.pipeline import Pipeline
    - Pipeline ham string listesini alır, TfidfVectorizer otomatik vektörize eder.
    - Adım adları MUTLAKA 'tfidf' ve 'clf' olsun (testler bunları kontrol eder).
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000)),
    ])


# 7. Modeli eğit
def train_model(pipe, X_train, y_train):
    """
    Pipeline'ı fit et ve döndür.

    Args:
        pipe: build_pipeline'dan dönen pipeline
        X_train: yorum metinleri (ham string'ler)
        y_train: target (0/1)

    Returns:
        Pipeline: fit edilmiş pipeline

    İpucu: pipe.fit(X_train, y_train); return pipe
    """
    pipe.fit(X_train, y_train)
    return pipe


# 8. Modeli değerlendir
def evaluate_model(pipe, X_test, y_test):
    """
    Test setinde tahmin yap, metrikleri hesapla.

    Returns:
        dict: {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'confusion_matrix': np.array (2x2)
        }

    İpucu:
    - y_pred = pipe.predict(X_test)
    - sklearn.metrics'ten: accuracy_score, precision_score, recall_score,
      f1_score, confusion_matrix
    - precision/recall/f1'de zero_division=0 kullan
    """
    y_pred = pipe.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }


# 9. Tek bir yorum için tahmin yap
def predict_review(pipe, text):
    """
    Bir yorum metni al, pos/neg tahmini + pozitif olasılığı döndür.

    Args:
        pipe: eğitilmiş pipeline
        text: str (bir film yorumu)

    Returns:
        dict: {
            'prediction': int (0=neg, 1=pos),
            'label': str ('pos' veya 'neg'),
            'pos_probability': float (pozitif olma olasılığı)
        }

    İpucu:
    - Pipeline tek string değil, liste bekler → [text] geç
    - proba = pipe.predict_proba([text])[0, 1]  # pozitif (pos) sınıf olasılığı
    - pred = int(pipe.predict([text])[0])
    """
    proba = float(pipe.predict_proba([text])[0, 1])
    pred = int(pipe.predict([text])[0])
    return {
        'prediction': pred,
        'label': 'pos' if pred == 1 else 'neg',
        'pos_probability': proba,
    }


# 10. En "pozitif" kelimeler
def top_positive_words(pipe, n=10):
    """
    Modelin pozitif yorumu en çok ayırt eden n kelimesini bul.

    LogisticRegression.coef_[0] → her kelimenin ağırlığı (katsayısı).
    Katsayı POZİTİF ve BÜYÜK olan kelimeler → modeli "pos" tarafına iten
    kelimeler (örn. 'great', 'excellent', 'perfect', 'wonderful' gibi bekleriz).

    Args:
        pipe: eğitilmiş pipeline
        n: kaç kelime döndürülecek (default 10)

    Returns:
        list: n adet kelime (str), en pozitiften aza doğru sıralı

    İpucu:
    - tfidf = pipe.named_steps['tfidf']; clf = pipe.named_steps['clf']
    - feature_names = tfidf.get_feature_names_out()
    - coefs = clf.coef_[0]
    - top_idx = np.argsort(coefs)[::-1][:n]  # en büyük n katsayı
    - [feature_names[i] for i in top_idx]
    """
    tfidf = pipe.named_steps['tfidf']
    clf = pipe.named_steps['clf']
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]
    top_idx = np.argsort(coefs)[::-1][:n]
    return [feature_names[i] for i in top_idx]


# 11. En "negatif" kelimeler
def top_negative_words(pipe, n=10):
    """
    top_positive_words'ün tersi — negatif yorumu en çok ayırt eden n kelime.
    Katsayısı en KÜÇÜK (en negatif) olan kelimeler (örn. 'bad', 'worst',
    'boring', 'waste' gibi bekleriz).

    Args:
        pipe: eğitilmiş pipeline
        n: kaç kelime (default 10)

    Returns:
        list: n adet kelime (str), en negatiften aza doğru sıralı

    İpucu:
    - coefs = clf.coef_[0]
    - top_idx = np.argsort(coefs)[:n]  # en küçük (en negatif) n katsayı
    """
    tfidf = pipe.named_steps['tfidf']
    clf = pipe.named_steps['clf']
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]
    top_idx = np.argsort(coefs)[:n]
    return [feature_names[i] for i in top_idx]



# 12. BoW vs TF-IDF karşılaştırması
def compare_bow_vs_tfidf(X_train, X_test, y_train, y_test):
    """
    İki pipeline eğit ve test F1'lerini karşılaştır:
    - bow_f1:   CountVectorizer(stop_words='english')  + LogisticRegression
    - tfidf_f1: TfidfVectorizer(stop_words='english')  + LogisticRegression

    Amaç: kelime SAYIMI (bag of words) ile TF-IDF ağırlıklandırmasının F1'e
    etkisini gözlemlemek (TF-IDF genelde biraz daha iyi sonuç verir).

    Returns:
        dict: {
            'bow_f1': float (F1),
            'tfidf_f1': float (F1)
        }

    İpucu:
    - from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    - İki ayrı Pipeline kur, her birini LogisticRegression(max_iter=1000) ile
      fit et, pipe.predict(X_test) → f1_score (zero_division=0).
    """
    results = {}
    vectorizers = [
        ('bow_f1', CountVectorizer(stop_words='english')),
        ('tfidf_f1', TfidfVectorizer(stop_words='english')),
    ]
    for key, vec in vectorizers:
        pipe = Pipeline([
            ('vec', vec),
            ('clf', LogisticRegression(max_iter=1000)),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results[key] = f1_score(y_test, y_pred, zero_division=0)
    return results


# 13. Tüm pipeline'ı uçtan uca çalıştır
def run_pipeline():
    """
    Uçtan uca akış:
    1. load_reviews (içinde fetch_reviews)
    2. explore_data (balance al)
    3. encode_labels
    4. split_data (X = df['text'], y = df['target'])
    5. build_pipeline → train_model
    6. evaluate_model (test F1)
    7. top_positive_words (en pozitif kelime)
    8. predict_review ile iki örnek yorum test et (biri pozitif, biri negatif)

    Örnek yorumlar (önerilen):
    - Pozitif: "An absolutely wonderful film, brilliant acting and a perfect
                script. I loved every minute."
    - Negatif: "A boring, terrible waste of time. The worst movie I have seen,
                awful and dull."

    Returns:
        dict: {
            'balance': float,
            'test_f1': float,
            'top_positive_word': str (top_positive_words listesinin ilki),
            'sample_pos_pred': dict (predict_review çıktısı),
            'sample_neg_pred': dict (predict_review çıktısı)
        }
    """
    # 1-2. Veri çek + yükle
    df = load_reviews()

    # 3. Keşif
    info = explore_data(df)

    # 4. Encode
    df = encode_labels(df)

    # 5. Split (X = yorum metni, y = target)
    X_train, X_test, y_train, y_test = split_data(df['text'], df['target'])

    # 6-7. Pipeline + eğit
    pipe = build_pipeline()
    pipe = train_model(pipe, X_train, y_train)

    # 8. Değerlendir
    metrics = evaluate_model(pipe, X_test, y_test)

    # 9-10. Örnek tahminler + en pozitif kelime
    pos_words = top_positive_words(pipe, n=10)
    sample_pos = predict_review(
        pipe,
        "An absolutely wonderful film, brilliant acting and a perfect script. I loved every minute.",
    )
    sample_neg = predict_review(
        pipe,
        "A boring, terrible waste of time. The worst movie I have seen, awful and dull.",
    )

    return {
        'balance': info['balance'],
        'test_f1': metrics['f1'],
        'top_positive_word': pos_words[0],
        'sample_pos_pred': sample_pos,
        'sample_neg_pred': sample_neg,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("📊 Pipeline Sonuçları:")
    print(f"  Pos/Neg dengesi    : {result['balance']:.2%}")
    print(f"  Test F1            : {result['test_f1']:.4f}")
    print(f"  En pozitif kelime  : {result['top_positive_word']}")
    print(f"  Pozitif örnek      : {result['sample_pos_pred']['label']} "
          f"(p={result['sample_pos_pred']['pos_probability']:.3f})")
    print(f"  Negatif örnek      : {result['sample_neg_pred']['label']} "
          f"(p={result['sample_neg_pred']['pos_probability']:.3f})")
