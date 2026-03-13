import asyncio
import threading

from transformers import pipeline

ENTITY_TYPE_MAP = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
    "MISC": "MISC",
}

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple",
                )
    return _model


def _run_ner(text: str) -> dict[str, list[str]]:
    if not text:
        return {}

    nlp = _get_model()
    results = nlp(text[:50_000])  # cap to bound processing time; BERT limit is 512 tokens

    entities: dict[str, list[str]] = {}
    for ent in results:
        entity_type = ENTITY_TYPE_MAP.get(ent["entity_group"])
        if entity_type is None:
            continue
        entity_text = ent["word"].strip()
        if len(entity_text) < 2:
            continue
        if entity_type not in entities:
            entities[entity_type] = []
        if entity_text not in entities[entity_type]:
            entities[entity_type].append(entity_text)

    return entities


class NERService:
    async def extract_entities(self, text: str) -> dict[str, list[str]]:
        if not text:
            return {}
        return await asyncio.to_thread(_run_ner, text)
