# Classification of a Heart with Hypertrophic Cardiomyopathy
This project was created as part of a recruitment task for the AI ​​med scientific group at AGH University, the data and idea are not my property.

Brief description:
Etapy działania programu:

1. Wczytanie danych

Program wczytuje dane z pliku task_data.csv przy użyciu biblioteki pandas.

Można sprawdzić nazwy kolumn - pierwsze kolumny to ID (numer zdjęcia, którego nie używamy w tej analizie) oraz Cardiomegaly - etykieta klasy: 0 (zdrowe serce) lub     1 (kardiomegalia). Pozostałe kolumny to różne pomiary z wartościami liczbowymi serca i płuc.

2. Czyszczenie i przygotowanie danych (Data Cleaning & Preprocessing)

Ponieważ dane zawierały wartości z przecinkami zamiast kropek, trzeba było:
	
	•	usunąć spacje z nazw kolumn (df.columns.str.strip()),
	
	•	zamienić przecinki na kropki i przekonwertować tekst na liczby (float),
	
	•	sprawdzić typy danych, aby wszystkie cechy były liczbowe.

3. Podział danych

Dane zostały podzielone na:
	
	•	zbiór treningowy (80%) - do nauki modelu,
	
	•	zbiór testowy (20%) - do sprawdzenia jego skuteczności.

Użyto random_state=42 dla powtarzalności wyniku.

4. Standaryzacja (normalizacja danych)

Cechy zostały wystandaryzowane przy użyciu StandardScaler(), przelicza wartości tak, by miały średnią = 0 i odchylenie standardowe = 1. Dzięki temu wszystkie cechy mają podobny zakres co jest ważne dla modeli takich jak regresja logistyczna. Bez tego model mógłby faworyzować duże wartości liczbowe.

5. Trenowanie modeli

Wybór modeli:

Wybrano modele, które są szybkie i dobrze działają przy małych zbiorach danych, takich jak tutaj. Drzewo decyzyjne jest łatwe do interpretacji, nie wymaga liniowych zależności ani wcześniejszego skalowania danych, a regresja logistyczna jest stabilnym modelem oraz dobrze radzi sobie w klasyfikacji binarnej (0, 1). 

Wybranie tych modeli pozwoliło na porównanie prostego, interpretowalnego modelu (drzewo) z bardziej statystycznym, probabilistycznym podejściem (regresja).

Komentarz do użycia klasyfikatorów:

Drzewo decyzyjne (Decision Tree) - polega na podejmowaniu decyzji poprzez kolejne pytania o wartość danej cechy. Użyty w programie parametr max_depth=3 ogranicza głębokość drzewa, co zapobiega przeuczeniu.

Regresja logistyczna (Logistic Regression) - estymuje prawdopodobieństwo, że pacjent jest chory.

6. Ewaluacja modeli

Accuracy (dokładność) - w obu modelach wyniosło tyle samo, co oznacza, że model poprawnie sklasyfikował 75% przypadków (6 z 8 obserwacji).

Macierz pomyłek (confusion matrix)

Oznacza:

	•	1 zdrowy pacjent poprawnie rozpoznany jako zdrowy (True Negative),
	
	•	1 zdrowy błędnie oznaczony jako chory (False Positive),
	
	•	1 chory błędnie oznaczony jako zdrowy (False Negative),
	
	•	5 chorych poprawnie rozpoznanych (True Positive).

Oznacza to, że model dobrze wykrywa przypadki choroby, ale czasami myli zdrowych pacjentów. 

W medycynie najniebezpieczniejszym błędem jest False Negative, kiedy chory pacjent zostanie zdiagnozowany jako zdrowy. Może to prowadzić do poważnych konsekwencji więc w analizie danych medycznych pożądane jest użycie metod, które w jak największym stopniu eliminują udział tego błędu. 

Precision, Recall, F1-score

Z raportu klasyfikacji:

	•	Precision (precyzja) - ile przypadków uznanych za chore naprawdę było chorych, czyli 0.83 = 83% pozytywnych prognoz było trafnych,
	
	•	Recall (czułość) - ile chorych pacjentów udało się poprawnie wykryć, czyli wykryto 83% faktycznie chorych,
	
	•	F1-score - średnia harmoniczna Precision i Recall, czyli 0.83 oznacza dobrą równowagę między wykrywaniem a dokładnością.

7. Walidacja krzyżowa (Cross-validation)

Zastosowano 5-krotną walidację krzyżową dla regresji logistycznej, co wskazuje na to, że model działa stabilnie i wyniki nie są przypadkowe - skuteczność waha się od 66% do 100% w zależności od podziału danych.

8. AUC (Area Under ROC Curve)

AUC (Area Under the Curve) mierzy zdolność modelu do rozróżniania klas. Im wyższy AUC, tym model lepiej oddziela pacjentów chorych od zdrowych. AUC = 1.0 oznacza idealny model, AUC = 0.5 - losowy. Oznacza to, że regresja logistyczna lepiej odróżnia chorych od zdrowych.

9. Wykresy ROC i Precision–Recall

Krzywa ROC (Receiver Operating Characteristic)

Tutaj regresja logistyczna ma wyraźnie większe pole pod krzywą (AUC = 0.83), więc jest dokładniejsza.

Krzywa Precision–Recall

Jest szczególnie przydatna, gdy dane są niezbalansowane, ponieważ skupia się na klasie pozytywnej (czyli chorzy pacjenci). Oba modele mają wysokie wartości obu miar (0.75–1.0). Regresja logistyczna utrzymuje lepszą stabilność precision przy różnych progach (linia wyżej). Oba modele są skuteczne, ale regresja logistyczna jest bardziej zrównoważona i pewniejsza w decyzjach.

10. Wniosek końcowy:

Oba modele osiągają podobną dokładność, ale regresja logistyczna ma wyższy AUC i lepiej rozróżnia klasy.Oznacza to, że przy nowych danych to właśnie regresja logistyczna byłaby lepszym wyborem jako model predykcyjny.

