"""Speculative translation/TTS worker; final pipeline still validates and exports."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jellydub.cache import cache_key, cache_lock
from jellydub.config import Config
from jellydub.models import Cue, JellyDubError
from jellydub.pipeline import get_translator, translator_options, tts_options
from jellydub.translate import dialogue_groups, translate_cues
from jellydub.tts import get_provider
from jellydub.tts.synthesize import synthesize_cues


def closed_prefix(cues, final=False):
    """Keep an unfinished singleton until the next cue determines its context."""
    if not cues or final:
        return cues, []
    last = dialogue_groups(cues)[-1]
    if len(last) == 1 and cues[-1].text.strip() and not cues[-1].text.rstrip().endswith(tuple('.!?…。！？:;')):
        return cues[:-1], cues[-1:]
    return cues, []


def main(stream):
    config = Config(**json.loads(next(stream)))
    config.validate()
    translator = provider = None
    pending = []
    speech_count = 0
    def process(cues):
        nonlocal translator, provider, speech_count
        if not cues:
            return
        if config.source_lang == "en":
            if translator is None:
                translator = get_translator(**translator_options(config))
            lock = config.cache_dir/"translation-locks"/(cache_key(translator.cache_id())+".lock")
            with cache_lock(lock):
                cues = translate_cues(cues, translator, config.cache_dir/"translations",
                                      strict_context_cache=True)
        if provider is None:
            provider = get_provider(config.tts, **tts_options(config))
        results, failures = synthesize_cues(cues, provider, config.cache_dir/"tts", workers=config.tts_workers)
        if failures:
            raise JellyDubError(f"Progressive TTS failed: {[(c.index, e) for c, e in failures]}")
        speech_count += len(results)
    try:
        for line in stream:
            pending.extend(Cue(**c) for c in json.loads(line))
            ready, pending = closed_prefix(pending)
            process(ready)
        process(pending)
        print(f"Progressive cache ready: {speech_count} speech cues", flush=True)
    finally:
        if translator is not None:
            translator.close()
        if provider is not None:
            provider.close()


if __name__ == "__main__":
    main(iter(sys.stdin))
