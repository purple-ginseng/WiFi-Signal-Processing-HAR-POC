"""Early stop for a timed collection session, shared by both GUIs."""
import threading
import time


class SessionStop:
    """Lets the Tk thread end a recording early and choose save or delete.

    The lock makes a Stop click and the timer running out mutually exclusive:
    a click either stops the session or has no effect, never half of each.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._start_ts = None
        self._recorded = None
        self._keep = True
        self._decided = threading.Event()

    def begin(self, start_ts):
        """Worker: recording started at start_ts; Stop now takes effect."""
        with self._lock:
            self._start_ts = start_ts
            self._recorded = None
            self._keep = True
            self._decided.clear()

    def request(self):
        """Tk thread: stop now. Seconds recorded, or None if not recording."""
        with self._lock:
            if self._start_ts is None or self._recorded is not None:
                return None
            self._recorded = time.time() - self._start_ts
            return self._recorded

    def requested(self):
        return self._recorded is not None

    def end(self):
        """Worker: recording over. Seconds recorded if stopped early, else None."""
        with self._lock:
            self._start_ts = None
            return self._recorded

    def decide(self, keep):
        """Tk thread: record the user's save (True) or delete (False) choice."""
        self._keep = keep
        self._decided.set()

    def keep(self):
        """Worker: block until the user has chosen, then return that choice."""
        self._decided.wait()
        return self._keep
