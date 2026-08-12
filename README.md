# 🛒 E-Commerce Review Analyzer

E-ticaret müşteri yorumlarını analiz etmek amacıyla geliştirilen **NLP tabanlı bir web uygulamasıdır**. Uygulama, müşteri yorumlarını metin işleme teknikleriyle analiz ederek yorumların **pozitif, negatif veya nötr** olduğunu belirler ve yorumlarda öne çıkan konu başlıklarını ortaya çıkarır.

Projede **BERT tabanlı sentiment analysis**, **topic modeling**, veri temizleme ve metin ön işleme yöntemleri kullanılmıştır. Analiz süreçleri bir pipeline yapısı içerisinde birleştirilerek web uygulaması üzerinden kullanılabilir hale getirilmiştir.

## ✨ Features

* 📝 E-ticaret müşteri yorumlarının metin ön işleme adımlarından geçirilmesi
* 🧠 **BERT tabanlı duygu analizi**
* 💬 Yorumların **Positive, Negative ve Neutral** olarak sınıflandırılması
* 🏷️ Müşteri yorumlarında öne çıkan konuların **topic modeling** ile belirlenmesi
* 📊 Eğitilen modellerin test ve değerlendirme süreçlerinin gerçekleştirilmesi
* 🔄 Veri işleme ve model süreçlerinin pipeline yapısında birleştirilmesi
* ⚡ FastAPI ile API servisinin oluşturulması
* 🌐 Analiz sonuçlarının web arayüzü üzerinden sunulması

## 🛠️ Technologies

- Python
- Hugging Face Transformers
- BERT
- Natural Language Processing (NLP)
- Sentiment Analysis
- Topic Modeling
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- FastAPI
- HTML

## 🔄 Project Workflow

Projenin temel çalışma akışı aşağıdaki şekilde tasarlanmıştır:

```text
Customer Reviews
       │
       ▼
Data Processing
       │
       ▼
Text Cleaning
       │
       ▼
BERT Sentiment Analysis
       │
       ├── Positive
       ├── Negative
       └── Neutral
       │
       ▼
Topic Modeling
       │
       ▼
Analysis Results
       │
       ▼
Pipeline / API
       │
       ▼
Web Application
```

## 🧠 Sentiment Analysis

Projenin temel bileşenlerinden biri müşteri yorumlarının duygu analizidir.

Bu amaçla **BERT tabanlı bir model** kullanılmıştır. Model, e-ticaret müşteri yorumlarını üç farklı sınıfa ayırmaktadır:

| Class        | Description               |
| ------------ | ------------------------- |
| **Positive** | Olumlu müşteri yorumları  |
| **Negative** | Olumsuz müşteri yorumları |
| **Neutral**  | Nötr müşteri yorumları    |

Modelin eğitim ve test süreçleri gerçekleştirilmiş, ardından model pipeline içerisinde kullanılmak üzere uygulamaya entegre edilmiştir.

Model ile ilgili eğitim ve test kodları `src/model/` klasöründe bulunmaktadır.

## 🏷️ Topic Modeling

Duygu analizine ek olarak, müşteri yorumlarında hangi konuların öne çıktığını belirlemek amacıyla **topic modeling** uygulanmıştır.

Bu aşama sayesinde yorumların yalnızca olumlu, olumsuz veya nötr olduğu değil, müşterilerin hangi konular hakkında yorum yaptığı da analiz edilebilmektedir.

Topic modeling ile ilgili uygulama `src/model/topic_model.py` içerisinde bulunmaktadır.

## 🧹 Data Processing

Modelleme öncesinde müşteri yorumlarının analiz için uygun hale getirilmesi amacıyla veri ve metin ön işleme adımları uygulanmıştır.

Bu kapsamda:

* 🧽 Veri temizleme
* ✍️ Metin temizleme
* 🗑️ Gereksiz içeriklerin düzenlenmesi
* 🔧 Modelleme için uygun veri formatının hazırlanması

gibi işlemler gerçekleştirilmiştir.

Veri işleme kodları:

```text
src/data_processing/
├── clean_data.py
└── clean_text.py
```

klasöründe bulunmaktadır.

## 📊 Model Training and Evaluation

Projede kullanılan BERT modelinin eğitim ve test süreçleri gerçekleştirilmiştir.

Model geliştirme sürecinde eğitim, test ve değerlendirme işlemleri ayrı kodlar halinde yapılandırılmıştır.

İlgili dosyalar:

```text
src/model/
├── bert_train.py
├── bert_test.py
├── train.py
├── test.py
├── grafik.py
└── topic_model.py
```

Model değerlendirme işlemleri ise `src/evalution/evaluation.py` üzerinden gerçekleştirilmektedir.

## ⚙️ Pipeline

Model ve veri işleme süreçlerinin uygulama içerisinde birlikte çalışabilmesi için bir **pipeline yapısı** oluşturulmuştur.

Pipeline içerisinde veri işleme, model kullanımı ve analiz sonuçlarının uygulamaya aktarılması gibi aşamalar bir araya getirilmiştir.

Pipeline yapısında ayrıca dashboard ve servis işlemleri için ayrı bir servis katmanı bulunmaktadır.

```text
src/pipeline/
├── pipeline.py
└── dashboard_service.py
```

## 🌐 Web Application

Projenin kullanıcı tarafından kullanılabilmesi için web tabanlı bir uygulama yapısı oluşturulmuştur.

API tarafında **FastAPI** kullanılmıştır. Web arayüzü ise HTML ile hazırlanmıştır.

Temel uygulama dosyaları:

```text
app/
└── index.html

src/api/
└── app.py
```

API, yorumların analiz süreçlerini pipeline ile birleştirerek elde edilen sonuçların uygulama üzerinden kullanılmasını sağlar.

## 📁 Project Structure

```text
E-Commerce Review Analyzer
│
├── app/
│   └── index.html
│
├── src/
│   ├── api/
│   │   └── app.py
│   │
│   ├── data_processing/
│   │   ├── clean_data.py
│   │   └── clean_text.py
│   │
│   ├── evalution/
│   │   └── evaluation.py
│   │
│   ├── model/
│   │   ├── bert_test.py
│   │   ├── bert_train.py
│   │   ├── grafik.py
│   │   ├── test.py
│   │   ├── topic_model.py
│   │   └── train.py
│   │
│   └── pipeline/
│       ├── dashboard_service.py
│       └── pipeline.py
│
├── data/
├── models/
├── artifacts/
├── .gitignore
└── README.md
```

## 📦 Dataset and Trained Models

Projede kullanılan datasetler ve eğitilmiş model dosyaları GitHub repository'sine dahil edilmemiştir.

* `data/` → Projede kullanılan datasetleri içerir.
* `models/` → Eğitilmiş modelleri ve tokenization dosyalarını içerir.
* `artifacts/` → Model süreçleri sonucunda oluşturulan artifact dosyalarını içerir.

Bu dosyalar boyutları nedeniyle `.gitignore` kullanılarak Git takibinin dışında bırakılmıştır.

Buna karşılık veri işleme, model eğitimi, test, değerlendirme ve pipeline süreçlerine ait **kaynak kodlar repository içerisinde bulunmaktadır.**

## 🚀 Installation

Projeyi çalıştırmadan önce gerekli Python paketlerini yükleyin:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Application

Gerekli bağımlılıklar kurulduktan sonra FastAPI uygulaması aşağıdaki komutla çalıştırılabilir:

```bash
uvicorn src.api.app:app --reload
```

Uygulama çalıştırıldıktan sonra API ve web arayüzü üzerinden yorum analiz süreçleri kullanılabilir.

## 🎯 Purpose

Bu projenin amacı, e-ticaret müşteri yorumlarını otomatik olarak analiz ederek **müşteri görüşlerinin duygu durumunu ve yorumlarda öne çıkan konu başlıklarını belirlemektir**.

BERT tabanlı duygu analizi ve topic modeling yöntemlerinin bir web uygulaması içerisinde bir araya getirilmesiyle, müşteri yorumlarından daha anlamlı ve kullanılabilir bilgiler elde edilmesi hedeflenmiştir.
