import numpy as np

from faceswap.schemas import VoiceConfig
from faceswap.speech import SpeechToSpeechProcessor
from faceswap.voice import (
    MonkVoiceProcessor,
    SounddeviceMonitor,
    _parse_pulse_sinks,
    _parse_pulse_source_outputs,
    _parse_pulse_sources,
    get_virtual_mic_route_status,
    live_voice_ducking_gain,
    play_monitor_test_tone,
    set_live_voice_ducking,
)


def test_monk_voice_processor_keeps_shape_and_limits_output():
    sample_rate = 48_000
    config = VoiceConfig(sample_rate=sample_rate, block_size=1024)
    processor = MonkVoiceProcessor(sample_rate, config)
    t = np.arange(config.block_size, dtype=np.float32) / sample_rate
    block = 0.12 * np.sin(2.0 * np.pi * 180.0 * t).astype(np.float32)

    output, level = processor.process(block[:, None])

    assert output.shape == block.shape
    assert level > 0
    assert np.isfinite(output).all()
    assert float(np.max(np.abs(output))) <= 0.98


def test_live_voice_ducking_gain_can_mute_and_restore():
    previous = set_live_voice_ducking(0.0)
    try:
        assert live_voice_ducking_gain() == 0.0
    finally:
        set_live_voice_ducking(previous)

    assert live_voice_ducking_gain() == previous


def test_monk_preset_masks_voice_even_when_depth_is_zero():
    sample_rate = 48_000
    config = VoiceConfig(
        preset="monk",
        sample_rate=sample_rate,
        block_size=1024,
        pitch_semitones=0.0,
        depth=0.0,
        reverb=0.0,
        echo=0.0,
        lowpass_hz=12_000.0,
        gain=1.0,
        noise_gate=0.0,
    )
    processor = MonkVoiceProcessor(sample_rate, config)
    t = np.arange(config.block_size, dtype=np.float32) / sample_rate
    block = 0.1 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)

    output, _level = processor.process(block[:, None])

    assert output.shape == block.shape
    assert np.isfinite(output).all()
    assert not np.allclose(output, block, atol=1e-3)


def test_speech_to_speech_fallback_masks_dry_voice_without_adapter(monkeypatch, tmp_path):
    from types import SimpleNamespace
    monkeypatch.setattr("faceswap.speech.get_settings", lambda: SimpleNamespace(models_dir=tmp_path))
    sample_rate = 48_000
    config = VoiceConfig(
        synthesis_mode="speech_to_speech",
        synthesis_target="monk",
        synthesis_quality="low_latency",
        sample_rate=sample_rate,
        block_size=512,
        gain=1.0,
        noise_gate=0.0,
    )
    processor = SpeechToSpeechProcessor(sample_rate, config)
    t = np.arange(config.block_size, dtype=np.float32) / sample_rate
    block = 0.1 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)

    output, level = processor.process(block[:, None])
    status = processor.status()
    processor.close()

    assert output.shape == block.shape
    assert level > 0
    assert np.isfinite(output).all()
    assert status.ready is True
    assert status.fallback is True
    assert "Seed-VC adapter unavailable" in (status.detail or "")
    assert status.latency_ms > 0
    assert not np.allclose(output, block, atol=1e-3)


def test_custom_preset_can_still_pass_dry_voice_with_depth_zero():
    sample_rate = 48_000
    config = VoiceConfig(
        preset="custom",
        sample_rate=sample_rate,
        block_size=1024,
        pitch_semitones=0.0,
        depth=0.0,
        reverb=0.0,
        echo=0.0,
        lowpass_hz=12_000.0,
        gain=1.0,
        noise_gate=0.0,
    )
    processor = MonkVoiceProcessor(sample_rate, config)
    t = np.arange(config.block_size, dtype=np.float32) / sample_rate
    block = 0.1 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)

    output, _level = processor.process(block[:, None])

    assert np.allclose(output, block, atol=1e-6)


def test_virtual_mic_route_status_is_safe_without_route():
    status = get_virtual_mic_route_status()

    assert status.sink_name == "faceswap_voice_sink"
    assert status.source_name == "faceswap_voice_mic"
    assert isinstance(status.sink_present, bool)
    assert isinstance(status.source_present, bool)


def test_sounddevice_monitor_writes_stereo_output():
    sounddevice = _FakeSounddevice()
    monitor = SounddeviceMonitor(
        sample_rate=48_000,
        block_size=8,
        output_device=3,
        volume=0.5,
        queue_blocks=4,
    )

    monitor.start(sounddevice)
    assert monitor.output_name == "Fake headphones"
    assert monitor.write(np.ones(8, dtype=np.float32))
    monitor.close()

    assert sounddevice.stream is not None
    assert sounddevice.stream.started is True
    assert sounddevice.stream.stopped is True
    assert len(sounddevice.stream.writes) == 1
    assert sounddevice.stream.writes[0].shape == (8, 2)
    assert np.allclose(sounddevice.stream.writes[0], 0.5)


def test_play_monitor_test_tone_writes_selected_output():
    sounddevice = _FakeSounddevice()

    result = play_monitor_test_tone(
        output_device=3,
        volume=0.4,
        duration_ms=100,
        frequency_hz=440.0,
        sounddevice=sounddevice,
    )

    assert result["ok"] is True
    assert result["output_name"] == "Fake headphones"
    assert sounddevice.stream is not None
    assert sounddevice.stream.started is True
    assert sounddevice.stream.stopped is True
    assert len(sounddevice.stream.writes) == 1
    assert sounddevice.stream.writes[0].ndim == 2
    assert sounddevice.stream.writes[0].shape[1] == 2
    assert np.max(np.abs(sounddevice.stream.writes[0])) > 0


def test_parse_pulse_sources_includes_usb_mono_and_skips_monitors():
    sources = _parse_pulse_sources(
        """
Source #707
    Name: alsa_output.usb-0c76_USB_Audio_Device-00.analog-stereo.monitor
    Description: Monitor of USB Audio Device Analog Stereo
    Sample Specification: s16le 2ch 48000Hz
    Monitor of Sink: alsa_output.usb-0c76_USB_Audio_Device-00.analog-stereo
Source #708
    Name: alsa_input.usb-0c76_USB_Audio_Device-00.mono-fallback
    Description: USB Audio Device Mono
    Sample Specification: s16le 1ch 48000Hz
    Monitor of Sink: n/a
Source #348
    Name: faceswap_voice_mic
    Description: FaceSwap_Voice_Mic
    Sample Specification: float32le 2ch 48000Hz
    Monitor of Sink: n/a
"""
    )

    assert len(sources) == 1
    assert sources[0].description == "USB Audio Device Mono"
    assert sources[0].name == "alsa_input.usb-0c76_USB_Audio_Device-00.mono-fallback"
    assert sources[0].channels == 1
    assert sources[0].sample_rate == 48_000
    assert sources[0].index < -100_000


def test_parse_pulse_sinks_includes_usb_output_and_skips_faceswap_sink():
    sinks = _parse_pulse_sinks(
        """
Sink #339
    Name: faceswap_voice_sink
    Description: FaceSwap_Voice_Sink
    Sample Specification: float32le 2ch 48000Hz
Sink #707
    Name: alsa_output.usb-0c76_USB_Audio_Device-00.analog-stereo
    Description: USB Audio Device Analog Stereo
    Sample Specification: s16le 2ch 48000Hz
"""
    )

    assert len(sinks) == 1
    assert sinks[0].description == "USB Audio Device Analog Stereo"
    assert sinks[0].name == "alsa_output.usb-0c76_USB_Audio_Device-00.analog-stereo"
    assert sinks[0].channels == 2
    assert sinks[0].sample_rate == 48_000
    assert sinks[0].index < -200_000


def test_parse_pulse_source_outputs_finds_obs_targeting_faceswap_mic():
    source_outputs = _parse_pulse_source_outputs(
        """
Source Output #644
    Source: 708
    Properties:
        application.name = "OBS"
        media.name = "Mic/Aux"
        target.object = "faceswap_voice_mic"
Source Output #666
    Source: 348
    Properties:
        application.name = "OBS"
        media.name = "Audio Input Capture (PulseAudio)"
        target.object = "faceswap_voice_mic"
"""
    )

    assert len(source_outputs) == 2
    assert source_outputs[0].index == 644
    assert source_outputs[0].source == 708
    assert source_outputs[0].target_object == "faceswap_voice_mic"
    assert source_outputs[0].application_name == "OBS"
    assert source_outputs[1].source == 348


class _FakeStream:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.writes: list[np.ndarray] = []

    def start(self) -> None:
        self.started = True

    def write(self, payload: np.ndarray) -> None:
        self.writes.append(payload.copy())

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        pass


class _FakeSounddevice:
    def __init__(self) -> None:
        self.stream: _FakeStream | None = None

    def query_devices(self, device=None, kind=None):
        assert device == 3
        assert kind == "output"
        return {"name": "Fake headphones", "max_output_channels": 2, "default_samplerate": 48_000}

    def OutputStream(self, **_kwargs):
        self.stream = _FakeStream()
        return self.stream
