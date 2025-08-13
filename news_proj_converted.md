```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
```


```python
news_data = pd.read_csv(r"C:\Users\DELL\news-article-categories.csv")
```


```python
news_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6877 entries, 0 to 6876
    Data columns (total 3 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   category  6877 non-null   object
     1   title     6877 non-null   object
     2   body      6872 non-null   object
    dtypes: object(3)
    memory usage: 161.3+ KB



```python
news_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>title</th>
      <th>body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ARTS &amp; CULTURE</td>
      <td>Modeling Agencies Enabled Sexual Predators For...</td>
      <td>In October 2017, Carolyn Kramer received a dis...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ARTS &amp; CULTURE</td>
      <td>Actor Jeff Hiller Talks “Bright Colors And Bol...</td>
      <td>This week I talked with actor Jeff Hiller abou...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ARTS &amp; CULTURE</td>
      <td>New Yorker Cover Puts Trump 'In The Hole' Afte...</td>
      <td>The New Yorker is taking on President Donald T...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ARTS &amp; CULTURE</td>
      <td>Man Surprises Girlfriend By Drawing Them In Di...</td>
      <td>Kellen Hickey, a 26-year-old who lives in Huds...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ARTS &amp; CULTURE</td>
      <td>This Artist Gives Renaissance-Style Sculptures...</td>
      <td>There’s something about combining the traditio...</td>
    </tr>
  </tbody>
</table>
</div>




```python
news_data.isnull().sum()
```




    category    0
    title       0
    body        5
    dtype: int64




```python
news_data = news_data.dropna(subset=['body'])
```


```python
news_data.isnull().sum()
```




    category    0
    title       0
    body        0
    dtype: int64




```python
label_encoder = LabelEncoder()
news_data['category_encoded'] = label_encoder.fit_transform(news_data['category'])
```


```python
X_train, X_test, y_train, y_test = train_test_split(news_data['body'],news_data['category_encoded'],test_size=0.2,random_state=42,stratify=news_data['category_encoded'])
```


```python
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
```


```python
print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)
```

    Training TF-IDF shape: (5497, 5000)
    Testing TF-IDF shape: (1375, 5000)



```python
nb_model = MultinomialNB()
```


```python
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
```


```python
nn_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=10, random_state=42)
```


```python
ensemble_model = VotingClassifier(
    estimators=[
        ('naive_bayes', nb_model),            
        ('logistic_regression', logistic_model),  
        ('neural_network', nn_model)              
    ], voting='soft'  )
```


```python
print("Training the ensemble model...")
ensemble_model.fit(X_train_tfidf, y_train)
```

    Training the ensemble model...


    C:\Users\DELL\anaconda3\Lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:686: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (10) reached and the optimization hasn't converged yet.
      warnings.warn(





<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>VotingClassifier(estimators=[(&#x27;naive_bayes&#x27;, MultinomialNB()),
                             (&#x27;logistic_regression&#x27;,
                              LogisticRegression(max_iter=1000,
                                                 random_state=42)),
                             (&#x27;neural_network&#x27;,
                              MLPClassifier(hidden_layer_sizes=(128,),
                                            max_iter=10, random_state=42))],
                 voting=&#x27;soft&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">VotingClassifier</label><div class="sk-toggleable__content"><pre>VotingClassifier(estimators=[(&#x27;naive_bayes&#x27;, MultinomialNB()),
                             (&#x27;logistic_regression&#x27;,
                              LogisticRegression(max_iter=1000,
                                                 random_state=42)),
                             (&#x27;neural_network&#x27;,
                              MLPClassifier(hidden_layer_sizes=(128,),
                                            max_iter=10, random_state=42))],
                 voting=&#x27;soft&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>naive_bayes</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">MultinomialNB</label><div class="sk-toggleable__content"><pre>MultinomialNB()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>logistic_regression</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=1000, random_state=42)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>neural_network</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">MLPClassifier</label><div class="sk-toggleable__content"><pre>MLPClassifier(hidden_layer_sizes=(128,), max_iter=10, random_state=42)</pre></div></div></div></div></div></div></div></div></div></div>




```python
print("Making predictions on the test data...")
y_pred = ensemble_model.predict(X_test_tfidf)
```

    Making predictions on the test data...



```python
accuracy = accuracy_score(y_test, y_pred)
print("Ensemble Model Accuracy:", accuracy)
```

    Ensemble Model Accuracy: 0.7941818181818182



```python
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
```

    
    Classification Report:
                     precision    recall  f1-score   support
    
    ARTS & CULTURE       0.76      0.91      0.83       201
          BUSINESS       0.72      0.68      0.70       100
            COMEDY       0.83      0.67      0.74        75
             CRIME       0.78      0.78      0.78        60
         EDUCATION       0.81      0.85      0.83        98
     ENTERTAINMENT       0.80      0.81      0.81       100
       ENVIRONMENT       0.82      0.81      0.81       100
             MEDIA       0.82      0.70      0.75        70
          POLITICS       0.74      0.78      0.76       100
          RELIGION       0.88      0.88      0.88       100
           SCIENCE       0.88      0.83      0.85        70
            SPORTS       0.88      0.88      0.88       100
              TECH       0.79      0.73      0.76       101
             WOMEN       0.70      0.65      0.67       100
    
          accuracy                           0.79      1375
         macro avg       0.80      0.78      0.79      1375
      weighted avg       0.80      0.79      0.79      1375
    



```python

```
