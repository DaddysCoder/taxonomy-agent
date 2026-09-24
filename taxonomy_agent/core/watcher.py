"""
Filesystem watcher using watchdog.

Monitors specified folders (e.g. ~/Downloads, ~/Desktop) for new files
and immediately classifies them. This is the "continuous" part — you never
have to manually run the organiser once the watcher is running.

Uses a small delay after detecting a file (SETTLE_SECONDS) to avoid
reading files that are still being written.
"""

from __future__ import annotations
import time
import threading
from pathlib import Path
from typing import Callable

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

SETTLE_SECONDS = 2.0   # wait after file appears before reading it
IGNORE_EXTENSIONS = {".tmp", ".part", ".crdownload", ".download"}
IGNORE_PREFIXES = {".", "~", "_"}


class TaxonomyEventHandler(FileSystemEventHandler):
    def __init__(self, classify_callback: Callable[[Path], None]):
        super().__init__()
        self.classify_callback = classify_callback
        self._pending: dict[str, threading.Timer] = {}

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._should_ignore(path):
            return
        # Debounce: cancel any pending timer for this path and restart
        if event.src_path in self._pending:
            self._pending[event.src_path].cancel()
        timer = threading.Timer(
            SETTLE_SECONDS,
            self._process,
            args=[path],
        )
        self._pending[event.src_path] = timer
        timer.start()

    def on_moved(self, event):
        """Also catch files moved INTO a watched directory."""
        if event.is_directory:
            return
        path = Path(event.dest_path)
        if not self._should_ignore(path):
            timer = threading.Timer(SETTLE_SECONDS, self._process, args=[path])
            timer.start()

    def _process(self, path: Path):
        self._pending.pop(str(path), None)
        if path.exists():
            self.classify_callback(path)

    def _should_ignore(self, path: Path) -> bool:
        name = path.name
        if any(name.startswith(p) for p in IGNORE_PREFIXES):
            return True
        if path.suffix.lower() in IGNORE_EXTENSIONS:
            return True
        return False


class FolderWatcher:
    def __init__(
        self,
        watch_dirs: list[Path],
        classify_callback: Callable[[Path], None],
    ):
        if not HAS_WATCHDOG:
            raise RuntimeError("watchdog not installed: pip install watchdog")
        self.watch_dirs = [Path(d) for d in watch_dirs]
        self.classify_callback = classify_callback
        self._observer = Observer()

    def start(self):
        handler = TaxonomyEventHandler(self.classify_callback)
        for directory in self.watch_dirs:
            if directory.exists():
                self._observer.schedule(handler, str(directory), recursive=False)
                print(f"[Watcher] Watching: {directory}")
            else:
                print(f"[Watcher] Directory not found, skipping: {directory}")
        self._observer.start()
        print("[Watcher] Running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self._observer.stop()
        self._observer.join()
        print("[Watcher] Stopped.")
