"""Reviewed speech overrides; subtitle text and timings remain untouched."""
from dataclasses import replace
import json
from pathlib import Path
import re
import warnings

from .models import JellyDubError


# Non-verbal tokens supported by the installed VieNeu v3 Turbo phonemizer.
_TAGS = {"cười", "cuoi", "chuckle", "thở dài", "tho dai", "sigh",
         "hắng giọng", "hang giong", "clear throat"}


_DIRECTION_FIELDS = {"emotion", "pace", "emphasis", "ending", "nonverbal"}


def _validate_direction(value):
    if not isinstance(value, dict) or set(value) - _DIRECTION_FIELDS:
        raise ValueError("direction must be an object with emotion, pace, emphasis, ending or nonverbal")
    for key, item in value.items():
        if key == "emphasis":
            if not isinstance(item, list) or any(not isinstance(v, str) or not v.strip() for v in item):
                raise ValueError("direction.emphasis must be an array of nonempty strings")
        elif not isinstance(item, str) or not item.strip():
            raise ValueError(f"direction.{key} must be a nonempty string")
    return dict(value)


def _notice(provider, cue_index, field, detail):
    notice = {"cue_index": cue_index, "field": field, "detail": detail}
    if not hasattr(provider, "speech_warnings"):
        provider.speech_warnings = []
    if notice not in provider.speech_warnings:
        provider.speech_warnings.append(notice)
        warnings.warn(f"TTS cue {cue_index} ({provider.name}), {field}: {detail}", stacklevel=3)


def _instructions(direction):
    labels = {"emotion": "Emotion", "pace": "Pace", "emphasis": "Emphasize these exact words",
              "ending": "Sentence ending", "nonverbal": "Nonverbal action"}
    return "; ".join(f"{labels[k]}: {', '.join(direction[k]) if k == 'emphasis' else direction[k]}"
                     for k in labels if direction.get(k))


def _direct_text(text, instruction):
    # VoxCPM2 expects a single leading parenthetical direction.
    existing = re.match(r"^\(([^()]*)\)", text)
    if existing:
        instruction += "; " + existing[1]
        text = text[existing.end():]
    instruction = instruction.replace("(", "（").replace(")", "）")
    return f"({instruction}){text}" if text.strip() else text


def _vieneu_direction(text, direction, provider, cue_index):
    for field in ("emotion", "emphasis", "ending"):
        if direction.get(field):
            _notice(provider, cue_index, field,
                    f"No direct VieNeu control for {direction[field]!r}; not applied. Reference audio may influence delivery.")
    pace = direction.get("pace", "").strip().lower()
    if pace:
        if pace in {"slow", "slowly", "chậm", "cham"}:
            text = re.sub(r"[,;](?=\s)", "…", text)
            text = re.sub(r"(?<=[^\W\d_])\.(?=\s+[^\W\d_])", "…", text)
            _notice(provider, cue_index, "pace", "Slow pace approximated with punctuation only; speaking rate is not controlled.")
        else:
            _notice(provider, cue_index, "pace", f"No direct VieNeu pace control for {pace!r}; not applied.")
    action = direction.get("nonverbal", "").strip().lower().rstrip(".! ")
    if action:
        for suffix in (" trước câu", " trước khi nói", " before speaking", " before the sentence"):
            if action.endswith(suffix):
                action = action[:-len(suffix)].strip()
                break
        tags = {"thở dài": "thở dài", "tho dai": "thở dài", "sigh": "thở dài",
                "cười": "cười", "cuoi": "cười", "chuckle": "cười",
                "hắng giọng": "hắng giọng", "hang giong": "hắng giọng", "clear throat": "hắng giọng"}
        tag = tags.get(action)
        if tag and text.strip():
            if not text.lstrip().startswith(f"[{tag}]"):
                text = f"[{tag}] {text}"
        elif not tag:
            _notice(provider, cue_index, "nonverbal", f"Unsupported VieNeu action {direction['nonverbal']!r}; not applied.")
    return text


def prepare_speech(cues, provider, plan_path=None):
    overrides = {}
    if plan_path:
        path = Path(plan_path).expanduser()
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError("expected an array of cue objects")
            for row in rows:
                if not isinstance(row, dict) or type(row.get("index")) is not int:
                    raise ValueError("each cue needs an integer index")
                if row["index"] in overrides:
                    raise ValueError(f"duplicate cue index {row['index']}")
                if set(row) - {"index", "text_sub", "text_tts", "emotion", "ref_audio", "direction"}:
                    raise ValueError("unknown speech cue fields")
                if any(not isinstance(v, str) for k, v in row.items() if k not in {"index", "direction"}):
                    raise ValueError("speech cue values must be strings")
                row = dict(row)
                if "direction" in row:
                    row["direction"] = _validate_direction(row["direction"])
                if "ref_audio" in row and not (provider.name == "gemini" and "direction" in row):
                    ref = Path(row["ref_audio"]).expanduser()
                    if not ref.is_absolute():
                        ref = path.parent / ref
                    if not ref.is_file():
                        raise ValueError(f"reference audio does not exist: {ref}")
                    row["ref_audio"] = str(ref.resolve())
                overrides[row["index"]] = row
        except (OSError, ValueError) as exc:
            raise JellyDubError(f"Cannot read speech plan {path}: {exc}") from exc
    result = {}
    checked_refs = set()
    if len({c.index for c in cues}) != len(cues):
        raise JellyDubError("Speech synthesis requires unique cue indices")
    for cue in cues:
        row = overrides.get(cue.index, {})
        if "text_sub" in row and row["text_sub"] != cue.text:
            raise JellyDubError(f"Speech plan subtitle mismatch at cue {cue.index}")
        item = replace(cue, **{k: v for k, v in row.items() if k not in {"index", "text_sub"}})
        direction = None
        if item.direction is not None:
            try:
                direction = _validate_direction(item.direction)
            except ValueError as exc:
                raise JellyDubError(f"Cue {cue.index}: {exc}") from exc
            if item.emotion:
                if direction.get("emotion", item.emotion) != item.emotion:
                    raise JellyDubError(f"Cue {cue.index}: conflicting emotion and direction.emotion")
                direction.setdefault("emotion", item.emotion)
        text = item.text if item.text_tts is None else item.text_tts
        tags = re.findall(r"\[([^\]]+)\]", text)
        if item.text_tts is not None and (any(t.strip().lower() not in _TAGS for t in tags)
                                         or "<|emotion_" in text):
            raise JellyDubError(f"Unsupported speech tag at cue {cue.index}")
        if tags and item.text_tts is not None and provider.name != "vieneu":
            raise JellyDubError("Non-verbal speech tags require VieNeu")
        speaker = item.speaker
        if item.ref_audio and provider.name == "gemini" and direction is not None:
            _notice(provider, cue.index, "ref_audio", "Gemini adapter uses named voices; reference audio is ignored.")
        elif item.ref_audio:
            if provider.name not in {"vieneu", "voxcpm2"}:
                raise JellyDubError("Per-cue reference audio requires VieNeu or VoxCPM2")
            from .cache import cache_key
            speaker = "speech:" + cache_key(item.speaker, item.ref_audio)
            from .ffmpeg import duration
            ref = Path(item.ref_audio)
            try:
                if ref not in checked_refs:
                    if not ref.is_file() or not 3 <= duration(ref) <= 8:
                        raise ValueError("expected a readable 3–8 second reference")
                    checked_refs.add(ref)
            except Exception as exc:
                raise JellyDubError(f"Invalid reference at cue {cue.index}: {exc}") from exc
            provider.speaker_refs[speaker] = ref
            if provider.name == "vieneu":
                provider.speaker_voices[speaker] = provider.speaker_voices.get(item.speaker, provider.voice)
        if direction is not None:
            instruction = _instructions(direction)
            if not hasattr(provider, "speech_directions"):
                provider.speech_directions = {}
            provider.speech_directions[cue.index] = {"requested": direction,
                "instructions": instruction if provider.name in {"gemini", "voxcpm2"} else None}
            if provider.name == "vieneu":
                text = _vieneu_direction(text, direction, provider, cue.index)
            elif provider.name == "voxcpm2" and instruction:
                text = _direct_text(text, instruction)
            elif provider.name == "gemini" and instruction:
                from .cache import cache_key
                speaker = "speech:direction:" + cache_key(item.speaker, instruction)
                provider.speaker_voices[speaker] = provider.speaker_voices.get(item.speaker, provider.voice)
                provider.speaker_styles[speaker] = "; ".join(filter(None, [provider.style, instruction]))
            elif provider.name not in {"vieneu", "voxcpm2", "gemini"} and instruction:
                _notice(provider, cue.index, "direction", "This provider does not support acting directions; not applied.")
        if direction is None and item.emotion and provider.name == "voxcpm2":
            text = f"({item.emotion}){text}"
        if direction is None and provider.name != "voxcpm2" and item.emotion and item.emotion not in {"neutral", "natural", "trung tính", "tự nhiên"} and not item.ref_audio and not tags:
            raise JellyDubError(f"Cue {cue.index}: emotion is metadata; provide ref_audio or a supported non-verbal tag")
        result[cue.index] = (text, speaker)
    return result
