# Struktura

### Projekt dzieli się logicznie na dwie główne części:

### 1. Przygotowanie danych i budowa modelu

`data_preparation_and_model_training.ipynb`

Notebook odpowiada za:

- wczytanie danych pomiarowych z plików CSV (IMU, GNSS, prędkość filtrowana),

- synchronizację danych z różnych źródeł na podstawie czasu,

- obliczenie cech opisujących dynamikę systemu (roll, drgania, rozbieżność prędkości),

- dyskretyzację cech ciągłych do stanów jakościowych (Low / Medium / High),

- ręczną definicję struktury sieci bayesowskiej oraz tablic prawdopodobieństw warunkowych (CPD),

- przeprowadzenie wnioskowania probabilistycznego dla danych historycznych,

- zapis wytrenowanego modelu do pliku integrity_network.pkl.

Efektem tej części jest gotowy model sieci bayesowskiej, przygotowany do użycia w systemie pokładowym.


### 2. Wnioskowanie w czasie rzeczywistym (ROS 2 node)

`bayesian_risk_analysis/bayesian_risk_analysis/risk_inference_node.py`

Druga część projektu to węzeł ROS 2, który wykorzystuje wcześniej wytrenowany model do bieżącej oceny stanu systemu łodzi.

- subskrybuje dane z czujników:

    - `/imu/data` – orientacja i przyspieszenia,

    - `/gnss_pose` – pozycja GNSS,

    - `/filter/velocity` – prędkość,

- oblicza w czasie rzeczywistym:

    - kąt roll,

    - całkowite przyspieszenie,

    - prędkość GNSS (na podstawie przyrostów pozycji),

    - rozbieżność pomiędzy prędkościami,

- Dokonuje dyskretyzacji sygnałów zgodnie z progami wyznaczonymi w etapie offline,

- wykonuje wnioskowanie probabilistyczne z użyciem sieci bayesowskiej,

- publikuje najbardziej prawdopodobny stan systemu na temat: `/integrity/system_status` (Int32).