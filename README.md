> Contiene i codici degli algoritmi in formato jupiter del paper OPS_SAT per esecuzione in locale
# Struttura
- File _dataset_generator_
  - File che genera i dati delle serie temporali del Dataset.
  - Il codice completo esegue diverse operazioni per trasformare un dataset di serie temporali in un dataset di caratteristiche e valutare eventuali anomalie.
- File _modeling_examples_
  - Codice di tutte le impementazioni degli algoritmi: supervised e unsupervised
  - Rockad: è presente un'implementazione rielaborata del codice preso dal paper (https://github.com/ml-and-vis/ROCKAD/tree/main), l'obbiettievo sarebbe trovare una giusta threshold per gli score che esso restituisce per un'equilibrata calibrazione della rilevazione delle anomalie
- File _requirement.txt_
  - File per generare l'ambiente .venv contenente i requisiti necessari per eseguire i modelli
  - Comprende, alcuni esempi:
    - pyod
    - sklearn
    - notebook
    - numpy
    - pandas
    - torch
- Cartella Rocket: contiene il codice fornito dal paper rocket e metriche aggiunte per la validazione dell'algoritmo
