# Speaker notes — „Shapley-Analyse eines MLP“
Speaker notes of the talk (German), extracted from `Shapley_MLP_Vortrag.pptx`. Slide numbers as in the deck; slides after the summary are reserve slides for questions.

## 1. Shapley-Analyse eines MLP (Titel)

Guten Tag, mein Name ist Arian Mohammadkhani. In meinem Vortrag geht es um die Shapley-Analyse eines Multi-Layer Perceptrons. Ich zeige, wie man die Vorhersagen eines neuronalen Netzes mit Shapley-Werten erklären kann – am Beispiel einer Hitzetag-Vorhersage.

## 2. Motivation: Warum erklären?

Ein echtes Beispiel aus meinem Testzeitraum: Für München sagt mein Modell am 29. Juni 2025: Morgen wird ein Hitzetag – mit 96 Prozent. Und es lag richtig. Aber warum sagt das Modell das? Ein neuronales Netz gibt nur eine Zahl aus, keine Begründung. Für Vertrauen und Fehlersuche brauchen wir Erklärungen. Genau das ist das Ziel von Explainable AI. Die Leitfrage heute: Welche Merkmale haben wie viel zur Vorhersage beigetragen? Bewusst nicht: Was hat den Hitzetag verursacht – das kann die Methode nicht beantworten. (Falls gefragt: Die 96 % sind die Modellausgabe. Weil Hitzetage im Training stärker gewichtet wurden, sind die Wahrscheinlichkeiten nach oben verschoben; die Entscheidungsschwelle liegt deshalb bei 90 %.)

## 3. Fahrplan

Der Vortrag hat fünf Teile, so nummeriert wie die Zwischenfolien: Zuerst Datensatz und Modell, dann die Mathematik der Shapley-Werte, dann zwei Näherungsverfahren, KernelSHAP und DeepSHAP. Danach zeige ich die Ergebnisse auf meinem Modell, und zum Schluss Fazit und Ausblick.

## 4. 01 Datensatz und Modell

—

## 5. Datensatz: DWD-Klimadaten

Als Datensatz verwende ich offizielle Tageswerte des Deutschen Wetterdienstes für fünf Städte. Die Aufgabe: Wird morgen ein Hitzetag, also eine Höchsttemperatur von mindestens 30 Grad? Dafür nutzt das Modell sieben Merkmale des heutigen Tages – fünf davon sind Temperaturen, das wird später wichtig. Das letzte Merkmal ist der Mittelwert der Höchsttemperaturen der drei Vortage. Die Daten sind zeitlich aufgeteilt: Das Modell lernt aus der Vergangenheit und wird auf 2024 und 2025 getestet. So sieht es nie Daten aus der Zukunft. Für München beginnen die Daten erst 1992.

## 6. Datensatzanalyse

Zwei Eigenschaften der Daten sind für später wichtig. Erstens sind Hitzetage selten: nur etwa drei Prozent der Tage, im Testzeitraum 3,7 Prozent. Ein Modell, das immer „kein Hitzetag“ sagt, hätte also schon 97 Prozent Accuracy – deshalb bewerte ich mit Precision, Recall und F1. Fast alle Hitzetage, 93 Prozent, liegen in Juni bis August. Zweitens sind die fünf Temperaturmerkmale stark korreliert, mit Werten zwischen 0,87 und 0,98; auch der Dampfdruck hängt eng mit der Temperatur zusammen. Das wird bei der Interpretation der Shapley-Werte eine Rolle spielen. Die Korrelation ist nur auf den Trainingsjahren berechnet.

## 7. MLP-Grundformel: eine Schicht

Eine Schicht eines MLP berechnet zuerst eine affine Transformation: Die Gewichtsmatrix W wird mit dem Eingabevektor multipliziert und der Bias b addiert. Jede Zeile von W enthält die Gewichte eines Neurons. Danach wird elementweise eine Aktivierungsfunktion angewendet, hier ReLU: Negative Werte werden zu null. Im Beispiel: Temperatur 28, Feuchte 60 ergibt die Werte 4 und 1. Die Ausgabe einer Schicht ist die Eingabe der nächsten.

## 8. Architektur des Modells

Mein Netz hat sieben Eingaben – die sieben Wettermerkmale –, eine verborgene Schicht mit 16 Neuronen und eine zweite mit 8 Neuronen. Zwei Begriffe brauchen wir im Rest des Vortrags. Das Embedding ist die Ausgabe der 8er-Schicht: acht Zahlen, mit denen das Netz einen Tag intern beschreibt. Der Logit z ist der Rohwert des letzten Neurons, bevor die Sigmoid-Funktion daraus eine Wahrscheinlichkeit macht. Ein Logit von null entspricht 50 Prozent. Das Netz hat 273 Parameter; weil Hitzetage selten sind, gewichte ich sie in der Verlustfunktion 32-mal stärker. Auf dem Testzeitraum 2024–25 findet es 90 von 135 Hitzetagen, also einen Recall von 0,67, bei einer Precision von 0,42 und einem F1 von 0,52. Damit ist es etwas besser als der Random Forest aus dem Ausgangsprojekt mit F1 0,50. MÖGLICHE FRAGE „Warum ein MLP?“: Das Seminarthema ist die Erklärung eines MLP. Ein MLP ist eine echte Blackbox, und DeepSHAP funktioniert nur für Netze. Ein Random Forest ließe sich mit TreeSHAP erklären – das ist ein Punkt für den Ausblick.

## 9. Warum ist ein MLP eine Blackbox?

Warum kann man ein MLP nicht einfach an den Gewichten ablesen? Nehmen wir ein einzelnes Neuron. An einem trockenen Tag bewirkt ein Grad mehr gar nichts – das Neuron bleibt aus. An einem feuchten Tag bewirkt dasselbe Grad plus eins. Die Wirkung der Temperatur hängt also von der Feuchte ab. Bei der logistischen Regression ändert ein Grad mehr den Logit immer um denselben Betrag – nur die Wahrscheinlichkeit ändert sich wegen der Sigmoid-Funktion unterschiedlich stark. Zwischenfazit: Im MLP wirken die Merkmale zusammen. Deshalb brauchen wir eine faire Methode, um eine Vorhersage auf die Merkmale aufzuteilen.

## 10. 02 Shapley-Werte

—

## 11. Kooperative Spieltheorie

Shapley-Werte stammen aus der kooperativen Spieltheorie. Dort arbeiten Spieler in Koalitionen zusammen und erzielen einen Gewinn. Die Frage ist: Wie verteilt man den Gewinn fair? Lloyd Shapley hat 1953 eine Antwort gegeben. Für Modelle übersetzen wir: Die Spieler sind die Merkmale, eine Koalition ist eine Menge bekannter Merkmale. Aber ein Netz kann nicht mit nur einem Teil der Merkmale rechnen. Deshalb ist f(S) genauer: Die fehlenden Merkmale werden mit Werten aus den Hintergrunddaten aufgefüllt – das ist eine Stichprobe normaler Tage –, und wir mitteln die Ausgaben.

## 12. Shapley-Wert am Beispiel

Ein Beispiel mit zwei Merkmalen. Vereinfacht nehme ich hier einen einzigen Referenztag statt ganzer Hintergrunddaten. Heute ist es 28 Grad bei 60 Prozent Feuchte, der Referenztag hat 25 Grad und 20 Prozent. Die Tabelle zeigt die Ausgabe für jede Koalition: Bekannte Merkmale haben den heutigen Wert, die anderen den Referenzwert. Kommt T zuerst, fügt T nichts hinzu und H den ganzen Rest, also 4. Kommt H zuerst, fügt H 1 hinzu und T 3. Keine Reihenfolge ist „die richtige“, also mitteln wir: T bekommt 1,5 und H 2,5. Zusammen ergibt das genau 4 – die gesamte Änderung der Ausgabe. Diese Zahlen kommen auf der DeepSHAP-Folie wieder.

## 13. Die Shapley-Formel

Das Mitteln über alle Reihenfolgen schreibt die Shapley-Formel kompakt auf. Wir summieren über alle Teilmengen S, die das Merkmal i nicht enthalten. Für jede Teilmenge berechnen wir den Grenzbeitrag: Wie stark ändert sich die Ausgabe, wenn i zu S hinzukommt? Diesen Beitrag gewichten wir mit dem Anteil der Reihenfolgen, in denen genau die Merkmale aus S vor i stehen. Daher kommen die Fakultäten: Sie zählen Reihenfolgen. Beispiel: Bei sieben Merkmalen und leerem S ist das Gewicht 0!·6!/7! = ein Siebtel – die Wahrscheinlichkeit, dass i zuerst kommt. Das Problem: Man braucht alle 2 hoch M Teilmengen. Bei sieben Merkmalen sind das 128, bei 50 Merkmalen etwa 10 hoch 15.

## 14. Die vier Axiome

Warum gerade Shapley-Werte? Weil sie vier Fairness-Regeln erfüllen. Effizienz: Die Werte addieren sich genau zur Differenz zwischen Vorhersage und Basiswert. Symmetrie: Zwei Merkmale mit immer gleichem Beitrag bekommen denselben Wert. Dummy: Ein Merkmal ohne Einfluss bekommt null. Additivität: Für die Summe zweier Modelle, etwa bei einem Ensemble, addieren sich die Werte. Shapley hat bewiesen, dass es genau eine Aufteilung gibt, die alle vier erfüllt. Achtung: Ein Wert von null heißt nicht automatisch, dass das Modell ein Merkmal ignoriert – positive und negative Effekte können sich aufheben. (Falls die Zeit knapp wird: diese Folie kürzen.)

## 15. Basiswert und Logit

Der Basiswert ist die mittlere Vorhersage des Modells auf den Hintergrunddaten – also was das Modell sagt, wenn es nichts über den heutigen Tag weiß. Die Shapley-Werte erklären den Weg vom Basiswert zur heutigen Vorhersage. Ich zeige das im Logit, weil sich die Beiträge dort genau addieren; die Sigmoid-Funktion würde sie verzerren. Das Beispiel ist der Münchner Tag von der Motivationsfolie: Vom Basiswert minus 13,25 führen Max.-Temperatur, Mittel-Temperatur, Dampfdruck und der Rest zum Logit 3,29, das sind 96 Prozent. Der Basiswert ist so niedrig, weil die Hintergrunddaten aus dem ganzen Jahr stammen und an Wintertagen der Logit bei etwa minus 25 liegt. Achtung: Den Basiswert nicht in Prozent umrechnen – der Mittelwert der Logits ist nicht der Logit der mittleren Wahrscheinlichkeit. Zwischenfazit: Shapley-Werte verteilen die Vorhersage fair, aber exakt braucht man 2 hoch M Auswertungen. Deshalb Näherungen.

## 16. 03 Näherungsverfahren

—

## 17. KernelSHAP

KernelSHAP löst das Laufzeitproblem durch Stichproben. Statt alle Koalitionen auszuwerten, zieht man zufällig einige. Jede Koalition ist ein Vektor aus Nullen und Einsen: eins heißt bekannt, null heißt ersetzt. Für jede Koalition wird das Modell ausgewertet – mit jedem Tag der Hintergrunddaten. Dann passt man eine gewichtete lineare Regression an; ihre Koeffizienten sind die geschätzten Shapley-Werte. Die Gewichte, der Shapley-Kernel, sind so gewählt, dass das Ergebnis mit allen Koalitionen exakt den Shapley-Werten entspricht. Vorteil: Es funktioniert für jedes Modell. Nachteil: Der Aufwand wächst mit Koalitionen mal Hintergrund für jeden Tag, und es kombiniert Merkmalswerte, als wären die Merkmale unabhängig.

## 18. DeepSHAP

DeepSHAP nutzt die Struktur des Netzes aus. Es basiert auf DeepLIFT: Für einen Referenzwert wird die Differenz zwischen Eingabe und Referenz Schicht für Schicht rückwärts durch das Netz verteilt. In linearen Schichten über die Gewichte, also Gewicht mal Delta. Bei ReLU wird proportional skaliert, mit dem Faktor m: Änderung nach ReLU geteilt durch Änderung vor ReLU. Am Beispiel-Neuron: Vor ReLU ändert sich z um 7 – 3 von der Temperatur, 4 von der Feuchte. ReLU lässt aber nur 4 durch, also m gleich vier Siebtel. DeepSHAP gibt der Temperatur 1,71 und der Feuchte 2,29. Exakt wären es 1,5 und 2,5. Die Summe 4 stimmt, aber die Aufteilung ist nur eine Näherung. Für viele Referenzen wird gemittelt.

## 19. KernelSHAP und DeepSHAP im Vergleich

Zusammengefasst: KernelSHAP schaut nicht ins Modell, ist dafür universell und wird mit allen Koalitionen exakt. Sein Aufwand wächst mit der Zahl der Koalitionen und der Hintergrunddaten. DeepSHAP nutzt die Schichten des Netzes und braucht nur einen Rückwärtsdurchlauf, liefert bei ReLU aber nur eine Näherung. Welche Methode auf meinem Modell schneller und genauer ist, zeigen die Ergebnisse – das Ergebnis ist nicht ganz so, wie man es erwarten würde.

## 20. 04 Ergebnisse

—

## 21. Exakte Werte als Referenz

Weil mein Modell nur sieben Merkmale hat, kann ich die Shapley-Werte exakt berechnen – mit eigenem Code über alle 128 Koalitionen, getestet an allen vier Axiomen. Damit habe ich eine Wahrheit, an der ich beide Näherungen messen kann. Jeder Punkt ist ein Merkmal an einem der 3650 Testtage: auf der Diagonale heißt identisch mit dem exakten Wert. KernelSHAP ist hier exakt, weil sein Standard-Budget von 2062 Stichproben alle 128 Koalitionen abdeckt. DeepSHAP weicht im Mittel um 0,1 ab, das sind 6,5 Prozent eines typischen Werts, höchstens um 1,12 – das ist die ReLU-Näherung von der DeepSHAP-Folie. Das wichtigste Merkmal stimmt an 94 Prozent der Tage mit dem exakten überein.

## 22. Zwei Testtage erklärt

Links der Münchner Tag, den das Modell richtig erkannt hat. Der Wasserfall startet beim Basiswert und zeigt, welche Merkmale den Logit nach oben oder unten schieben: vor allem Max.-Temperatur, Mittel-Temperatur und Dampfdruck. Rechts der sicherste Fehlalarm im Testzeitraum: Frankfurt am 2. Juli 2025, 38,4 Grad. Das Modell sagte mit über 99 Prozent einen Hitzetag voraus – am nächsten Tag waren es aber nur 27,5 Grad, ein plötzlicher Wetterumschwung. Hier zeigt sich der Nutzen der Methode: Wir sehen, warum das Modell falsch lag. Es reagiert fast nur auf die Hitze von heute; Informationen über einen Umschwung wie Luftdruck oder Wind hat es nicht. Die Werte addieren sich exakt zum Logit – der Effizienz-Check ist erfüllt.

## 23. Testzeitraum: globale Sicht

Der Beeswarm-Plot zeigt alle Testtage: jeder Punkt ein Tag, die Farbe der Merkmalswert, rot hoch, blau niedrig. Hohe Temperaturen schieben den Logit nach oben. Interessant ist die Feuchte: Hohe Feuchte senkt das Risiko – heiße Tage sind bei uns meist trocken. Wichtig: Fünf meiner sieben Merkmale sind Temperaturen, die stark korreliert sind. Ihr Beitrag verteilt sich deshalb auf alle fünf, manchmal sogar mit entgegengesetzten Vorzeichen, und jedes einzelne wirkt kleiner, als die Temperatur insgesamt ist. Wegen der Additivität darf man die Werte addieren: Zusammen kommen die Temperaturen auf 6,2, das stärkste Einzelmerkmal nur auf 2,4.

## 24. Ergebnisse: Embeddings

Bis jetzt waren die Merkmale die Spieler für den Logit. Jetzt schaue ich ins Netz, in die Schicht mit den acht Embedding-Neuronen, und zwar auf zwei Arten. Links sind die Merkmale die Spieler, erklärt wird jedes Neuron einzeln: Worauf reagiert es? Rechts sind die acht Neuronen selbst die Spieler für den Logit: Welche Neuronen entscheiden? Die letzte Schicht ist linear, deshalb ist der Wert hier einfach Gewicht mal Abweichung vom Mittel, und auch DeepSHAP ist dort exakt. Das Ergebnis: Die vier wichtigsten Neuronen haben negative Gewichte und sind an über 90 Prozent aller Tage aktiv – „Normalwetter“-Neuronen. Vor Hitzetagen schalten die beiden stärksten nur noch an etwa einem Viertel der Tage ein. Das Netz erkennt Hitze also vor allem daran, dass diese Neuronen abschalten. Die drei Neuronen mit positivem Gewicht sind vor fast jedem Hitzetag aktiv, tragen aber wenig bei; eines wird praktisch nicht genutzt. Links sieht man: Die wichtigen Neuronen reagieren auf Temperatur, Dampfdruck und Feuchte. Embedding-Dimensionen haben keine Namen – mit dieser Analyse kann man ihnen eine Bedeutung geben.

## 25. Laufzeitanalyse

Ich habe die Rechenzeit in drei Richtungen gemessen. Links: Alle drei Verfahren wachsen linear mit der Anzahl erklärter Tage. Mitte: Exakt und DeepSHAP wachsen linear mit den Hintergrunddaten, bei KernelSHAP dominiert bis etwa 100 Tage der feste Aufwand für Stichproben und Regression. Überraschend: Bei meinen sieben Merkmalen ist die exakte Berechnung sogar am schnellsten. Rechts der Grund, warum es trotzdem Näherungen braucht: Auf Zufallsnetzen mit mehr Eingaben verdoppelt sich der exakte Aufwand mit jedem Merkmal und überholt KernelSHAP zwischen 12 und 14 Merkmalen. DeepSHAP ist praktisch unabhängig von der Merkmalsanzahl. Die einfache Aussage „KernelSHAP langsam, DeepSHAP schnell“ stimmt für dieses kleine Modell also nicht – entscheidend ist die Zahl der Merkmale. Gemessen auf einem MacBook Air mit M1.

## 26. 05 Fazit

—

## 27. Grenzen

Drei Grenzen. Erstens: Bei abhängigen Merkmalen wie unseren Temperaturen werden einzelne Werte durch Werte anderer Tage ersetzt – so entstehen Tage, die es nie gibt, etwa ein hohes Maximum bei niedrigem Mittelwert. Das betrifft nicht nur KernelSHAP, sondern auch meine exakte Berechnung, weil alle drei dasselbe Spiel verwenden. Das Modell wird dort ausgewertet, wo es nie trainiert wurde. Zweitens hängt das Ergebnis stark von den Hintergrunddaten ab: Mit Sommertagen statt Tagen aus dem ganzen Jahr sinkt der Beitrag der Max.-Temperatur am Münchner Tag von 5,9 auf 2,5. Eine Erklärung muss deshalb immer sagen, im Vergleich zu welchen Tagen. Drittens zeigen Shapley-Werte, worauf das Modell reagiert – nicht, was in der Realität die Ursache ist.

## 28. Fazit und Ausblick

Mein Fazit: Shapley-Werte verteilen eine Vorhersage fair auf die Merkmale. Weil mein Modell nur sieben Merkmale hat, konnte ich sie exakt berechnen und die Näherungen daran überprüfen: KernelSHAP war exakt, DeepSHAP wich im Mittel um 6,5 Prozent ab – und bei sieben Merkmalen war die exakte Berechnung sogar am schnellsten. Inhaltlich dominiert die Temperatur als Gruppe, und im Netz erkennt man Hitze vor allem daran, dass „Normalwetter“-Neuronen abschalten. Offene Punkte: Abhängigkeiten berücksichtigen, etwa mit bedingten Shapley-Werten nach Aas und anderen; Luftdruck und Wind als Merkmale, damit Wetterumschwünge wie beim Frankfurter Fehlalarm erkennbar werden; und das MLP mit einem Random Forest und TreeSHAP vergleichen.

## 29. Vielen Dank für Ihre Aufmerksamkeit!

—

## 30. Quellen (1/3)

—

## 31. Quellen (2/3)

—

## 32. Quellen (3/3)

—

## 33. Backup

—

## 34. Backup: exakte Werte vs. Näherung

Der Münchner Hitzetag aus dem Test: meine exakte Berechnung über alle 128 Koalitionen im Vergleich zu KernelSHAP und DeepSHAP. KernelSHAP stimmt auf zwei Nachkommastellen überein. DeepSHAP verteilt etwas anders, am deutlichsten bei Min.-Temperatur und Vortags-Max. Die Summe ist bei allen drei gleich: 16,54 = Logit 3,29 minus Basiswert −13,25. Die Effizienz gilt also auch für die Näherung – nur die Aufteilung unterscheidet sich.

## 35. Backup: SHAP-Eigenschaften und Axiome

Lundberg und Lee formulieren drei Eigenschaften. Local accuracy entspricht der Effizienz. Missingness: Ein Merkmal, das in der vereinfachten Eingabe fehlt, bekommt null – verwandt mit dem Dummy-Axiom. Consistency: Wenn ein Modell sich so ändert, dass ein Merkmal mehr beiträgt, darf sein Wert nicht sinken. Young hat 1985 gezeigt, dass diese Monotonie zusammen mit Effizienz und Symmetrie den Shapley-Wert ebenfalls eindeutig festlegt – sie ersetzt also die Additivität. Darauf stützt sich Theorem 1 bei Lundberg und Lee.

## 36. Backup: KernelSHAP-Stichproben

Wie viele Koalitionen hat KernelSHAP ausgewertet? Der Standard der shap-Bibliothek ist 2 mal M plus 2048 Stichproben. Bei sieben Merkmalen gibt es nur 128 Koalitionen – KernelSHAP rechnet dann exakt. Um die Näherung sichtbar zu machen, habe ich die Anzahl bewusst kleiner gesetzt, auf 300 Testtagen: Mit 16 Koalitionen liegt die mittlere Abweichung bei 0,13, mit 64 bei 0,02, mit allen 126 bei null. Wichtig: Dabei habe ich die Lasso-Merkmalsauswahl der Bibliothek ausgeschaltet, die sie bei wenigen Stichproben sonst automatisch zuschaltet – sonst misst man ein anderes Verfahren.

## 37. Backup: Trainingsdetails

Trainingsdetails: binäre Kreuzentropie, Hitzetage mit pos_weight 32 gewichtet, Adam mit Lernrate 0,001, Batchgröße 256, Early Stopping auf dem Validierungsverlust – das beste Modell war in Epoche 40. Die Eingaben sind mit Mittelwert und Standardabweichung der Trainingsjahre standardisiert. Die Entscheidungsschwelle ist das beste F1 auf den Validierungsjahren, 0,9 – so hoch, weil die Gewichtung alle Wahrscheinlichkeiten nach oben verschiebt. Eigenes Experiment ohne Gewichtung: Schwelle 0,3, Recall 0,56 statt 0,67, F1 0,50 statt 0,52, PR-AUC sogar leicht höher (0,45 statt 0,43). Die Gewichtung verschiebt also vor allem die Schwelle; die Rangfolge der Tage ist ähnlich.

## 38. Backup: Wahl der Hintergrunddaten

Derselbe Münchner Tag, dasselbe Modell, derselbe Logit 3,29 – nur andere Hintergrunddaten. Links Tage aus dem ganzen Jahr: Basiswert minus 13,25, Max.-Temperatur plus 5,90. Rechts nur Sommertage: Basiswert minus 1,23, Max.-Temperatur nur plus 2,47. Ein Shapley-Wert beantwortet immer die Frage „im Vergleich wozu?“. Im Vergleich zum ganzen Jahr ist ein 31-Grad-Tag extrem, im Vergleich zum Sommer nur etwas überdurchschnittlich. Die Rangfolge der Merkmale bleibt ähnlich. Auch die zufällige Auswahl der 100 Hintergrundtage spielt eine Rolle: Zehn verschiedene Ziehungen ändern die Werte im Mittel um etwa 12 Prozent.
