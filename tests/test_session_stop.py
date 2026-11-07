"""bfmtool.session_stop: the Stop click vs timer race and the save/delete hand-off."""
import threading
import time

from bfmtool.session_stop import SessionStop


def test_request_reports_seconds_recorded():
    stop = SessionStop()
    stop.begin(time.time() - 40)
    assert 39 < stop.request() < 41
    assert stop.requested()


def test_request_before_recording_does_nothing():
    stop = SessionStop()
    assert stop.request() is None
    assert not stop.requested()


def test_second_click_does_nothing():
    stop = SessionStop()
    stop.begin(time.time())
    assert stop.request() is not None
    assert stop.request() is None


def test_click_after_timer_ended_does_nothing():
    stop = SessionStop()
    stop.begin(time.time())
    assert stop.end() is None
    assert stop.request() is None


def test_end_reports_an_early_stop_and_can_repeat():
    stop = SessionStop()
    stop.begin(time.time() - 5)
    recorded = stop.request()
    assert stop.end() == recorded
    assert stop.end() == recorded


def test_begin_clears_the_previous_session():
    stop = SessionStop()
    stop.begin(time.time())
    stop.request()
    stop.decide(False)
    stop.begin(time.time())
    assert not stop.requested() and stop.end() is None


def test_keep_blocks_until_decided():
    stop = SessionStop()
    stop.begin(time.time())
    stop.request()
    got = []
    worker = threading.Thread(target=lambda: got.append(stop.keep()), daemon=True)
    worker.start()
    worker.join(0.1)
    assert worker.is_alive() and got == []
    stop.decide(False)
    worker.join(2)
    assert got == [False]


def test_click_and_timer_racing_always_agree():
    """A click that took effect must be seen by the worker, in every interleaving."""
    for _ in range(500):
        stop = SessionStop()
        stop.begin(time.time())
        barrier = threading.Barrier(2)
        clicked, ended = [], []

        def click():
            barrier.wait()
            clicked.append(stop.request())

        def timer():
            barrier.wait()
            ended.append(stop.end())

        threads = [threading.Thread(target=f) for f in (click, timer)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert (clicked[0] is None) == (ended[0] is None)
