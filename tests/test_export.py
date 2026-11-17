import hashlib
import subprocess

import pytest

from jellydub.export import export_mp3, export_mp4, mux_mkv, output_paths
from jellydub.ffmpeg import probe, run_ffmpeg
from jellydub.models import JellyDubError


def test_paths(tmp_path):
    video = tmp_path / 'Movie [2026].mkv'
    paths = output_paths(video)
    assert paths['audio'].name == 'Movie [2026].AI-Dub.vi.mp3'
    assert paths['srt'].name == 'Movie [2026].vi.srt'
    assert paths['mp4'].name == 'Movie [2026].AI-Dub.vi.mp4'
    assert output_paths(video, tmp_path / 'out')['mkv'].parent == tmp_path / 'out'
    with pytest.raises(JellyDubError): output_paths(video, label='../bad')


def test_export_and_mux_preserve_source(tmp_path):
    source = tmp_path / 'movie [1].mkv'
    run_ffmpeg(['-f', 'lavfi', '-i', 'testsrc=size=32x32:rate=10:duration=1', '-f', 'lavfi', '-i', 'sine=duration=1', '-c:v', 'ffv1', '-c:a', 'eac3', '-ac', '6', str(source)])
    checksum = hashlib.sha256(source.read_bytes()).digest()
    dub = tmp_path / 'dub.wav'
    run_ffmpeg(['-f', 'lavfi', '-i', 'sine=frequency=800:duration=1', str(dub)])
    subtitle = tmp_path / 'movie.vi.srt'
    subtitle.write_text('1\n00:00:00,000 --> 00:00:01,000\nXin chào\n', encoding='utf-8')
    mp3 = tmp_path / 'out' / 'movie.AI-Dub.vi.mp3'
    export_mp3(dub, mp3)
    assert probe(mp3)['streams'][0]['channels'] == 2
    output = tmp_path / 'dubbed.mkv'
    mux_mkv(source, mp3, subtitle, output)
    streams = probe(output)['streams']
    assert streams[0]['codec_name'] == 'ffv1'
    tracks = [s for s in streams if s['codec_type'] == 'audio']
    assert tracks[0]['codec_name'] == 'eac3'
    assert tracks[0]['channels'] == 6
    assert tracks[0]['disposition']['default'] == 0
    assert tracks[1]['disposition']['default'] == 1
    assert tracks[1]['tags']['language'] == 'vie'
    assert tracks[1]['tags']['title'] == 'Tiếng Việt (AI Dub)'
    captions = [s for s in streams if s['codec_type'] == 'subtitle']
    assert len(captions) == 1
    assert captions[0]['codec_name'] == 'subrip'
    assert captions[0]['disposition']['default'] == 1
    assert captions[0]['tags']['language'] == 'vie'
    assert captions[0]['tags']['title'] == 'Phụ đề tiếng Việt'
    assert hashlib.sha256(source.read_bytes()).digest() == checksum
    assert not list(tmp_path.rglob('*.part.*'))
    with pytest.raises(JellyDubError): mux_mkv(source, mp3, subtitle, source)


def test_mp4_burns_vietnamese_subtitles_and_selects_dub_audio(tmp_path):
    source = tmp_path / 'movie[1].mp4'
    run_ffmpeg(['-f', 'lavfi', '-i', 'color=c=black:s=1920x1080:r=10:d=1',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)])
    checksum = hashlib.sha256(source.read_bytes()).digest()
    dub = tmp_path / 'voice.wav'
    run_ffmpeg(['-f', 'lavfi', '-i', 'sine=frequency=800:duration=1', str(dub)])
    subtitle = tmp_path / 'subtitles [vi].srt'
    subtitle.write_text(
        '1\n00:00:00,000 --> 00:00:01,000\n'
        'Chào mừng đến với Pragmatic. Hôm nay, tôi sẽ giải thích cách phân tích chênh lệch '
        'giá giữa các nhà cung cấp, còn gọi là so sánh các phương án báo giá RFQ.\n',
        encoding='utf-8',
    )
    output = tmp_path / 'output.mp4'

    export_mp4(source, dub, subtitle, output)

    streams = probe(output)['streams']
    assert next(s for s in streams if s['codec_type'] == 'video')['codec_name'] == 'h264'
    audio = next(s for s in streams if s['codec_type'] == 'audio')
    assert audio['codec_name'] == 'aac'
    assert audio['tags']['language'] == 'vie'
    assert not [s for s in streams if s['codec_type'] == 'subtitle']
    frame = subprocess.run([
        'ffmpeg', '-v', 'error', '-ss', '0.5', '-i', str(output),
        '-frames:v', '1', '-vf', 'crop=1920:400:0:680,format=gray',
        '-f', 'rawvideo', '-pix_fmt', 'gray', '-',
    ], capture_output=True, check=True).stdout
    visible_rows = [
        row for row in range(400)
        if any(value > 160 for value in frame[row * 1920:(row + 1) * 1920])
    ]
    assert visible_rows
    assert visible_rows[-1] - visible_rows[0] < 240
    assert hashlib.sha256(source.read_bytes()).digest() == checksum


def test_atomic_failure_keeps_previous_export(tmp_path, monkeypatch):
    import jellydub.export as module
    output = tmp_path / 'existing.mp3'
    output.write_bytes(b'previous export')
    def fail(args):
        from pathlib import Path
        Path(args[-1]).write_bytes(b'partial')
        raise JellyDubError('encoder failed')
    monkeypatch.setattr(module, 'run_ffmpeg', fail)
    with pytest.raises(JellyDubError, match='encoder failed'):
        export_mp3(tmp_path / 'input.wav', output)
    assert output.read_bytes() == b'previous export'
    assert not (tmp_path / 'existing.part.mp3').exists()
