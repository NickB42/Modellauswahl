# LLM Response Time Benchmark

Dieses Repository enthält den Code und die Ergebnisse meiner Bachelorarbeit zur Leistungsmessung verschiedener Large Language Models (LLMs).

## Projektübersicht

Das Projekt misst und analysiert die Antwortzeiten und Token-Generierungsgeschwindigkeiten von verschiedenen kommerziellen LLMs, darunter Modelle von OpenAI, Anthropic, Google und Mistral AI.

## Dateien

- `responseMultiThread.py`: Hauptskript für die Durchführung der Benchmarks
- `Analysis.py`: Generiert statistische Auswertungen der Ergebnisse
- `Graph.py`: Visualisiert die Benchmark-Ergebnisse
- `results Ordner`: Beeinhaltet die einzelnen Ergebnisse für jedes Modell.
- `combined_results.json`: Beeinhaltet die kombinierten Ergebnisse für alle Modelle. Dient als Basis der Auswertungen.

## Verwendung

### Installation

Das Projekt verwendet Python-Abhängigkeiten, die in der `requirements.txt` Datei aufgeführt sind. So richten Sie die Umgebung ein:

```bash
# Virtuelle Umgebung erstellen
python -m venv venv

# Virtuelle Umgebung aktivieren
# Unter Windows:
venv\Scripts\activate
# Unter macOS/Linux:
source venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### Benchmarks ausführen
Benchmarks können aufgrund der Verwendung SAP internen Architektur nicht von Dritten ausgeführt werden.
```bash
python responseMultiThread.py
```

### Ergebnisse analysieren
Dieses Skript kann ausgeführt werden.
```bash
python Analysis.py
# Wenn aufgefordert, Pfad zur JSON-Ergebnisdatei eingeben
# Hier: combined_results.json
```

### Ergebnisse visualisieren
Dieses Skript kann ausgeführt werden.
```bash
python Graph.py
# Wenn aufgefordert, Pfad zur JSON-Ergebnisdatei eingeben
# Hier: combined_results.json
```