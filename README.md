# ensemble-methods
*Experimentos de análise comparativa entre métodos isolados e combinados*

## 🔹 Visão geral

Este repositório reúne três projetos/notebooks de machine learning que exploram distintos cenários de predição, com foco especial em **ensemble methods**: comparar modelos clássicos (isolados) vs. combinar modelos via *voting* / *stacking*. O objetivo é:  
- aplicar técnicas supervised learning sem redes neurais profundas;  
- avaliar desempenho de modelos individuais;  
- comparar com ensemble (voting e stacking);  
- observar ganhos (ou não) com a combinação de modelos;  
- documentar boas práticas de ML (pré-processamento, tuning, validação, ensemble).

---

## 📁 Estrutura do repositório

| Arquivo / Notebook | Descrição |
|------------------|-----------|
| `Facial_Recognition_with_Supervised_Learning.ipynb` | Experimento de reconhecimento facial (“pessoa X ou não”) usando features PCA. Testa modelos clássicos + voting + stacking. |
| `ml_projeto_ensemble_class_good_book.ipynb` | Projeto genérico de classificação (base “bom livro/exemplo de livro didático”) para comparar modelos clássicos e ensembles. |
| `predicting_movie_rental_durations.ipynb` | Regressão (ou regressão → classificação/região?) de duração de locação de filmes — explora predição de variáveis contínuas, possivelmente com ensembles (ou baseline de regressão). |
| `README.md` | Documentação principal |

---

## 🧠 Métodos e técnicas utilizados

Em diferentes notebooks, foram usados os seguintes métodos:

### ✅ Modelos individuais / baselines  
- **LogisticRegression** — regressão logística para classificação.  
- **SVC** (SVM) — classificação com margem, usando kernel(s) configuráveis.  
- **KNeighborsClassifier** (KNN) — classificação baseada em similaridade/distância no espaço de features.  
- Para problemas de regressão (quando aplicável): regressão linear ou similar (dependendo do notebook).  

### 🧩 Técnicas de ensemble  
- **VotingClassifier** — ensemble “soft voting” para classificação: combina probabilidades (ou scores) de múltiplos classificadores e decide pela classe com maior média.  
- **StackingClassifier** — stacking (empilhamento): os modelos base geram predições que servem como features para um “meta-classificador” (no seu caso, geralmente LogisticRegression).  

### 🔧 Pré-processamento & tuning  
- Aplicação de **PCA** para redução de dimensionalidade (especialmente no notebook de reconhecimento facial).  
- Uso de **RandomizedSearchCV** para ajustes de hiperparâmetros (C, kernel, número de vizinhos, pesos, etc.).  
- Splits de treino/teste para validação da generalização.  

---

## 📊 O que foi testado / Métricas & Avaliação  

Para cada experimento foram avaliadas — quando cabível — métricas como:  
- F1-score (para classificação) — via `f1_score`.  
- AUC / ROC (quando aplicável).  
- Comparação das performances dos modelos individuais vs. ensemble (voting / stacking).  

Além disso, busca-se observar:  
- Se ensemble supera modelos individuais;  
- Em quais cenários (tipo de dados / distribuição / número de features) ensembles trazem ganho ou não;  
- E quais trade-offs aparecem (complexidade, risco de overfitting, custo computacional).  

---

## ✅ O que foi aprendido / Conclusões parciais

- Ensembles via **VotingClassifier (soft voting)** tendem a dar ganhos consistentes quando os modelos base têm erros distintos (complementares).  
- **Stacking** — quando implementado corretamente (com predições out-of-fold para meta) — pode superar o voting, mas exige cuidado para evitar *data leakage*.  
- Pré-processamento e redução de dimensionalidade (como PCA) + tuning de hiperparâmetros são fundamentais para extrair bom desempenho de modelos clássicos.  
- Modelos simples (LogisticRegression, SVM, KNN) ainda são bastante úteis quando combinados, mesmo sem redes neurais / deep learning — especialmente em domínios com features estruturadas ou extraídas via PCA.  
- Em problemas com muitos dados ou alta dimensionalidade, a combinação de métodos e validação cuidadosa melhora estabilidade e generalização.  

---

## 🎯 Quando usar este repositório / Para quem serve

Este repositório é útil para:  
- quem quer aprender e comparar **métodos clássicos de ML + ensembles**;  
- quem está em contextos onde **deep learning não é viável** — por restrições computacionais, de dados ou de interpretabilidade;  
- estudantes ou profissionais que querem ver **práticas de ML end-to-end**: pré-processamento, tuning, ensemble, avaliação;  
- servir como base para adaptar para outros problemas (outros datasets de classificação / regressão).  

---

## 🚀 Como rodar / Pré-requisitos

1. Tenha instalado Python (versão ≥ 3.8) e bibliotecas usuals: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn` (se usar visualizações), etc.  
2. Clone este repositório:  
   ```bash
   git clone https://github.com/analluvias/ensemble-methods.git  
   cd ensemble-methods  
