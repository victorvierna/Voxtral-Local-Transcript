import json

from voxtray.recordings import RecordingArtifactStore


def test_recording_artifact_store_saves_audio_and_metadata(tmp_path):
    store = RecordingArtifactStore(base_dir=tmp_path)

    saved = store.save(
        source="microphone",
        model_id="mistralai/Voxtral-Mini-4B-Realtime-2602",
        sample_rate=16000,
        chunk_ms=40,
        pcm16_audio=b"\x01\x00" * 1600,
        raw_text="hola mundo",
        normalized_text="Hola mundo.",
        status="success",
        requested_segment_max_seconds=90,
        effective_segment_max_seconds=24.576,
        segment_texts=["hola mundo"],
        diagnostics={"completion_status": "complete", "segments": [{"index": 1}]},
    )

    assert saved.directory.exists()
    assert saved.audio_path.exists()
    assert saved.metadata_path.exists()
    assert saved.audio_path.stat().st_size > 44

    metadata = json.loads(saved.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "success"
    assert metadata["raw_text"] == "hola mundo"
    assert metadata["normalized_text"] == "Hola mundo."
    assert metadata["audio_duration_seconds"] == 0.1
    assert metadata["segment_texts"] == ["hola mundo"]
    assert metadata["diagnostics"]["completion_status"] == "complete"
    assert metadata["diagnostics"]["segments"] == [{"index": 1}]
