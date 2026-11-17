"""Generate a small Gemini sample without persisting the API key."""
import getpass
import json
import os
from pathlib import Path
import time
import wave
from jellydub.tts.gemini import GeminiTTS
from jellydub.tts.base import TTSError

if __name__ == '__main__':
    key = getpass.getpass('Gemini API key (not saved): ')
    os.environ['GEMINI_API_KEY'] = key
    key = None
    target = Path('artifacts/gemini-tts/sample.vi.wav')
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        GeminiTTS(style='Speak natural Vietnamese, clearly and expressively.').synthesize(
            'Chúng ta đang thử giọng đọc tiếng Việt mới. Mục tiêu là xử lý video nhanh hơn, đồng thời giữ giọng nói rõ ràng và tự nhiên.', target)
        with wave.open(str(target)) as audio:
            result = dict(status='success', path=str(target), seconds=audio.getnframes()/audio.getframerate(), sample_rate=audio.getframerate())
    except TTSError as exc:
        result = dict(status='failed', error=str(exc))
    finally:
        os.environ.pop('GEMINI_API_KEY', None)
    result['elapsed_seconds'] = round(time.monotonic()-started, 3)
    (target.parent/'sample-result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
