"""
local_session_store.py - Drop-in replacement for redis.StrictRedis on localhost.
Thread-safe in-memory dict with TTL expiry. Supports get/set/setex/ping/delete.
"""

import threading
import time


class LocalSessionStore:
    def __init__(self, **kwargs):
        self._data = {}
        self._expiry = {}
        self._lock = threading.Lock()

    def ping(self):
        return True

    def setex(self, key, ttl_seconds, value):
        with self._lock:
            self._data[key] = value
            self._expiry[key] = time.monotonic() + ttl_seconds

    def set(self, key, value):
        with self._lock:
            self._data[key] = value
            self._expiry.pop(key, None)

    def get(self, key):
        with self._lock:
            exp = self._expiry.get(key)
            if exp is not None and time.monotonic() > exp:
                self._data.pop(key, None)
                self._expiry.pop(key, None)
                return None
            return self._data.get(key)

    def delete(self, *keys):
        with self._lock:
            for key in keys:
                self._data.pop(key, None)
                self._expiry.pop(key, None)
