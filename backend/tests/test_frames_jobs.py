import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from faceswap.frames import LatestCapture
from faceswap.gpu_jobs import GpuCoordinator, JobCancelled
from faceswap.lifecycle import SessionBusy
from faceswap.avatar import AvatarStudio
from faceswap.schemas import AvatarProfile, AvatarRenderRequest


def test_capture_keeps_only_latest_frame_and_releases():
    finished = threading.Event()
    class Capture:
        released = False
        index = 0
        def read(self):
            self.index += 1
            if self.index > 20:
                finished.wait(2)
            return True, np.full((2, 2, 3), min(self.index, 255), np.uint8)
        def release(self):
            self.released = True
    capture = Capture()
    reader = LatestCapture(capture).start()
    deadline = time.monotonic() + 1
    while reader.sequence < 20 and time.monotonic() < deadline:
        threading.Event().wait(.001)
    sequence, timestamp, frame = reader.next(0)
    assert sequence == 20 and timestamp > 0
    assert frame[0, 0, 0] == 20
    reader.stop_event.set()
    finished.set()
    reader.close()
    assert capture.released and not reader.thread.is_alive()


def test_capture_owns_frames_returned_to_processing_thread():
    release = threading.Event()
    shared = np.zeros((2, 2, 3), np.uint8)
    class ReusingCapture:
        def read(self):
            shared.fill(7)
            release.wait(1)
            return True, shared
        def release(self):
            pass
    reader = LatestCapture(ReusingCapture()).start()
    release.set()
    deadline = time.monotonic() + 1
    while reader.sequence == 0 and time.monotonic() < deadline:
        time.sleep(.001)
    reader.stop_event.set()
    _, _, frame = reader.next(0)
    shared.fill(99)
    assert np.all(frame == 7)
    reader.close()


def test_gpu_admission_excludes_offline_and_live():
    gate = GpuCoordinator()
    owner = object()
    gate.enter_live(owner)
    with pytest.raises(JobCancelled):
        with gate.render(lambda: True):
            pass
    gate.leave_live(owner)
    with gate.render(lambda: False):
        with pytest.raises(SessionBusy):
            gate.enter_live(owner)
    gate.enter_live(owner)
    gate.leave_live(owner)


def test_avatar_queue_cancellation_and_restart_recovery(tmp_path, monkeypatch):
    studio = AvatarStudio(tmp_path)
    entered, release = threading.Event(), threading.Event()
    profile = AvatarProfile(id='p', display_name='Test', target_image_path='target',
        voice_reference_path='voice', created_at='2026-09-19T00:00:00Z', consent_confirmed=True)
    monkeypatch.setattr(studio, 'get_profile', lambda _: profile)
    def run(job_id, request):
        entered.set()
        release.wait(2)
        studio._check_cancelled(job_id)
        studio._update_job(job_id, status='succeeded')
    monkeypatch.setattr(studio, '_run_job', run)
    request = AvatarRenderRequest(profile_id='p', text='Test render')
    first = studio.submit_render(request)
    assert entered.wait(1)
    second = studio.submit_render(request)
    assert studio.get_job(second.id).status == 'queued'
    assert studio.cancel(second.id).status == 'cancelled'
    assert studio.cancel(first.id).status == 'cancelling'
    release.set()
    studio._queue.join()
    assert studio.get_job(first.id).status == 'cancelled'
    studio.close()
    interrupted = first.model_copy(update={'id': 'b'*32, 'status': 'running'})
    studio._store_job(interrupted)
    recovered = AvatarStudio(tmp_path)
    assert recovered.get_job(interrupted.id).status == 'failed'
    recovered.close()
