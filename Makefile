.PHONY: install install-train generate dedup train quantize eval clean help

VENV := .venv
PY   := $(VENV)/bin/python

help:
	@echo "code-distill pipeline:"
	@echo "  make install    create .venv + deps (see requirements.txt for Unsloth note)"
	@echo "  make generate   Phase 1 - teacher loaded in LM Studio -> raw.jsonl"
	@echo "  make dedup      Phase 2 - clean + split -> clean.jsonl, heldout.jsonl"
	@echo "  make train      Phase 3 - UNLOAD teacher first! QLoRA -> merged model"
	@echo "  make quantize   Phase 4 - GGUF + Q5_K_M"
	@echo "  make eval       Phase 5 - load student in LM Studio, then run"
	@echo "  make clean      remove generated artifacts (keeps raw.jsonl + .venv)"

# Arch ships only current Python; uv fetches a managed 3.12 (Unsloth-compatible).
$(PY):
	uv venv --python 3.12 $(VENV)

install: $(PY)
	VIRTUAL_ENV=$(VENV) uv pip install -r requirements.txt
	@echo ">> Base deps installed in $(VENV)."
	@echo ">> For training, also run:  make install-train"

install-train: $(PY)
	VIRTUAL_ENV=$(VENV) uv pip install unsloth

generate: $(PY)
	@echo ">> Teacher ($(shell $(PY) -c 'import config;print(config.TEACHER_MODEL)')) must be loaded in LM Studio on :1234"
	$(PY) gen.py

dedup: $(PY)
	$(PY) dedup.py

train: $(PY)
	@echo ">> Did you UNLOAD the teacher in LM Studio? Both won't fit in 16GB."
	$(PY) train.py

quantize: $(PY)
	VENV=$(VENV) bash quantize.sh

eval: $(PY)
	@echo ">> Load your quantized student in LM Studio, set STUDENT_ID if needed."
	$(PY) eval.py

clean:
	rm -rf out gguf $(shell $(PY) -c 'import config;print(config.MERGED_DIR)' 2>/dev/null) \
	       clean.jsonl heldout.jsonl __pycache__
	@echo ">> kept raw.jsonl and .venv (delete manually for a full reset)"
