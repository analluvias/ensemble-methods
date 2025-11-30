# Applied Machine Learning Portfolio

Este repositório consolida três projetos práticos de Machine Learning focados na comparação entre modelos isolados e **Métodos de Ensemble** (Stacking e Voting). O objetivo principal foi investigar se a combinação de modelos preditivos supera o desempenho de algoritmos individuais bem ajustados em diferentes domínios: Regressão, Processamento de Linguagem Natural (NLP) e Classificação de Imagens (via PCA).

## Tecnologias e Ferramentas

* **Linguagem:** Python
* **Bibliotecas Principais:** Pandas, Scikit-learn, Numpy, Seaborn, NLTK.
* **Conceitos Chave:** Pipelines, Feature Engineering, Hyperparameter Tuning (RandomizedSearchCV), Voting Classifiers, Stacking Regressors/Classifiers.

---

## Projetos

### 1. Previsão de Duração de Aluguel de Filmes 
**Arquivo:** `predicting_movie_rental_durations.py`

**Objetivo:** Prever o número de dias que um cliente ficará com um DVD alugado (Regressão).

**Técnicas e Implementação:**
* **Feature Engineering:**
    * Tratamento de strings com Regex para criar *dummy variables* a partir da coluna `special_features` (Trailers, Deleted Scenes, etc.).
    * Cálculo de aritmética de datas (`return_date` - `rental_date`) para definir o target.
* **Modelagem:**
    * Comparação entre **Lasso**, **Ridge**, **Linear Regression** e **Gradient Boosting Regressor**.
    * Uso de `StandardScaler` e `ColumnTransformer` dentro de pipelines.
* **Ensemble:**
    * Implementação de um **Stacking Regressor** utilizando Lasso e Ridge na primeira camada e Linear Regression como meta-model.

**Aprendizado / Resultados:**
* O **Gradient Boosting** e o **Stacking Regressor** superaram os modelos lineares simples.
* Demonstrou como modelos de árvores e técnicas de ensemble capturam melhor as não-linearidades do comportamento do consumidor do que regressões puras.

---

### 2. Classificação de Sucesso de Livros (Good Reads) 
**Arquivo:** `ml_projeto_ensemble_class_good_book.py`

**Objetivo:** Classificar se um livro é "Popular" ou "Impopular" com base em metadados e reviews textuais.

**Técnicas e Implementação:**
* **NLP (Processamento de Linguagem Natural):**
    * Limpeza de texto: remoção de pontuação (Regex), conversão para minúsculas e remoção de **Stopwords** (NLTK).
    * Vetorização: Uso intensivo de **TF-IDF** (Term Frequency-Inverse Document Frequency) aplicado separadamente a múltiplas colunas textuais (título, descrição, autores).
* **Feature Engineering Numérica:**
    * Criação da métrica `fraction-helpfulness` para tratar a utilidade das reviews.
* **Modelagem & Ensemble:**
    * Modelos base: KNN, Regressão Logística e Árvore de Decisão.
    * Otimização de hiperparâmetros com `RandomizedSearchCV`.
    * Comparação com **Soft Voting** e **Stacking Classifier**.

**Aprendizado / Resultados:**
* **Insight Crítico:** Nem sempre o Ensemble vence. Neste caso específico, a **Regressão Logística** bem ajustada (L2 penalty) superou ou empatou com os métodos de ensemble mais complexos.
* Isso reforçou a importância do princípio da parcimônia (Occam's Razor) em ML: se um modelo simples resolve bem, ele é preferível devido ao menor custo computacional.

---

### 3. Reconhecimento Facial (Arnold Schwarzenegger) 
**Arquivo:** `facial_recognition_with_supervised_learning.py`

**Objetivo:** Classificação binária para identificar se uma imagem pertence ao Arnold Schwarzenegger ou não, utilizando o dataset *Labeled Faces in the Wild*.

**Técnicas e Implementação:**
* **Dados:** Utilização de dados pré-processados via **PCA** (Principal Component Analysis) para redução de dimensionalidade.
* **Modelagem:**
    * SVM (Support Vector Machine), KNN e Regressão Logística.
    * Ajuste fino de `class_weight='balanced'` para lidar com desbalanceamento de classes.
* **Ensemble:**
    * **Soft Voting Classifier** combinando as probabilidades dos três modelos base.
    * **Stacking Classifier** usando Regressão Logística como meta-estimador.

**Aprendizado / Resultados:**
* O **Voting Classifier** foi o grande vencedor, superando todos os modelos individuais com margem significativa.
* O Stacking teve performance similar ao melhor modelo individual (LogReg), indicando que para este dataset, a "democracia" das probabilidades (Voting) funcionou melhor que a re-aprendizagem (Stacking).

---

## Como Executar

1. Clone o repositório.
2. Instale as dependências:
   ```bash
   pip install pandas scikit-learn seaborn numpy nltk regex
