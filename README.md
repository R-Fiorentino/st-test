#  Search API

API FastAPI che legge il catalogo `data/products.csv` e restituisce prodotti rilevanti in JSON.

## Endpoint

- `GET /health`
- `GET /search?q=outfit eleganti da ufficio&limit=8`

## Avvio locale

```bash
python -m venv .venv
source .venv/bin/activate  # su Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Apri poi:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/search?q=outfit%20eleganti%20da%20ufficio&limit=8`

## Deploy su Render

1. Carica questa cartella su GitHub.
2. Su Render crea un nuovo Web Service collegando il repo.
3. Usa:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Deploy.

## Deploy su Railway

1. Carica questa cartella su GitHub.
2. In Railway: `New Project` → `Deploy from GitHub repo`.
3. Espone poi un dominio pubblico dal pannello Networking.

## Note

- Per iniziare, il CSV è incluso nel progetto: non serve hostare l'Excel separatamente.
- Se in futuro vuoi aggiornare il feed spesso, puoi spostare il CSV in Supabase Storage e farlo leggere all'API.
- La ricerca usa tag, descrizioni e categorie, con espansione semantica per query come `ufficio`, `elegante`, `outfit`, `estate`, `travel`.
