import sqlite3
import os
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event
import time

DB_FOLDER = r"C:\placas\reconocimientoDePlacas\proyectoPlacasModularizado\database"
DB_PATH = Path(DB_FOLDER) / "placas.sqlite"
os.makedirs(DB_FOLDER, exist_ok=True)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS detecciones_placas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_ts TEXT NOT NULL,
    plate TEXT NOT NULL,
    rango TEXT,
    tecnica TEXT
);
"""
PRAGMAS_AND_INDEXES = [
    "PRAGMA journal_mode = WAL;",
    "PRAGMA synchronous = NORMAL;",
    "PRAGMA foreign_keys = ON;",
    "CREATE INDEX IF NOT EXISTS idx_plate ON detecciones_placas(plate);",
    "CREATE INDEX IF NOT EXISTS idx_detection_ts ON detecciones_placas(detection_ts);"
]

colaEscritura = Queue(maxsize=2000)
_stopEvent = Event()
_workerThread = None
BATCH_SIZE = 20
FLUSH_INTERVAL = 1.0

def iniciarBaseDatos():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.cursor()
    try:
        cur.execute(CREATE_TABLE_SQL)
        for stmt in PRAGMAS_AND_INDEXES:
            try:
                cur.execute(stmt)
            except:
                pass
        conn.commit()
        print("✅ Base de datos inicializada:", DB_PATH)
    finally:
        conn.close()

def _workerLoop():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=True)
    try:
        conn.execute("PRAGMA busy_timeout = 5000;")
    except:
        pass
    cur = conn.cursor()
    buffer = []
    lastFlush = time.time()

    while not _stopEvent.is_set() or not colaEscritura.empty() or buffer:
        try:
            item = colaEscritura.get(timeout=0.25)
            buffer.append(item)
            colaEscritura.task_done()
        except Empty:
            pass

        now = time.time()
        if (len(buffer) >= BATCH_SIZE) or (buffer and (now - lastFlush) >= FLUSH_INTERVAL) or (_stopEvent.is_set() and buffer):
            try:
                cur.executemany("INSERT INTO detecciones_placas (detection_ts, plate, rango, tecnica) VALUES (?, ?, ?, ?);", buffer)
                conn.commit()
            except Exception as e:
                print(f"[DB WORKER] Error: {e}")
            finally:
                buffer.clear()
                lastFlush = time.time()
    conn.close()

def iniciarWorker():
    global _workerThread
    if _workerThread is None or not _workerThread.is_alive():
        _stopEvent.clear()
        _workerThread = Thread(target=_workerLoop, daemon=True)
        _workerThread.start()
        print("✅ Worker de DB iniciado.")

def detenerWorker(waitSeconds=2.0):
    _stopEvent.set()
    if _workerThread is not None:
        _workerThread.join(timeout=waitSeconds)
        print("✅ Worker de DB detenido.")

