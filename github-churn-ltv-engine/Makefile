.PHONY: install db pipeline api test up down
install:   ; pip install -r requirements.txt
db:        ; docker compose up -d db
pipeline:  ; python -m src.pipeline
api:       ; uvicorn api.main:app --reload --port 8000
test:      ; pytest -q
up:        ; docker compose up --build
down:      ; docker compose down
