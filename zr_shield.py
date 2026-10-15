"""
ZR-SHIELD 2.0
Protection visuelle de l'écran contre les regards indiscrets.

BMM Systems, projet ZYRHA OS
Auteure : Bella Marie Mathilde
Master 2 Cybersécurité et Blockchain, Keyce Informatique, Yaoundé

Principe
    La webcam surveille l'environnement. Le propriétaire est reconnu par
    son visage (modèle SFace) et par sa distance de travail habituelle.
    Dès qu'une autre personne apparaît dans le champ, l'écran est flouté
    sauf une petite zone nette qui suit le curseur de saisie ou la souris.
    Le propriétaire, à 45 cm, continue de lire et d'écrire dans cette zone ;
    la personne placée derrière lui ne voit qu'une image dégradée.
"""

import sys
import os
import re
import json
import time
import hmac
import ctypes
import hashlib
import secrets
import datetime
import math

import numpy as np
import cv2

from PyQt5.QtCore import (
    Qt, QThread, QTimer, QObject, QPoint, QPointF, QRect, QRectF, QSize, QVariantAnimation, QEasingCurve, pyqtSignal,
)
from PyQt5.QtGui import (
    QPainter, QColor, QImage, QPixmap, QIcon, QPainterPath, QPen, QPalette,
    QCursor, QLinearGradient, QFont, QFontMetrics,
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAbstractButton, QDialog, QWidget, QFrame, QLabel, QPushButton,
    QLineEdit, QSlider, QSpinBox, QCheckBox, QRadioButton, QButtonGroup,
    QVBoxLayout, QHBoxLayout, QGridLayout, QStackedWidget, QListWidget,
    QListWidgetItem, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QProgressBar, QMessageBox, QSystemTrayIcon, QMenu,
    QFileDialog, QSizePolicy,
)

APP_NAME = "ZR-SHIELD"
APP_VERSION = "2.0"
IS_WINDOWS = sys.platform == "win32"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
YUNET_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.]{3,24}$")

DEFAULT_CONFIG = {
    "distance_cm": 45,          # distance de travail déclarée lors de la calibration
    "ref_face_ratio": 0.0,      # largeur du visage / largeur de l'image à cette distance
    "range_m": 3.0,             # au delà, les visages sont ignorés
    "match_threshold": 0.36,    # similarité cosinus SFace minimale pour "c'est moi"
    "trigger_ms": 400,
    "release_ms": 700,          # suivi d'un fondu de 1,5 s
    "effect": "flou",           # flou | pixels | voile
    "strength": 7,
    "lens_enabled": True,
    "lens_size": "moyen",       # petit | moyen | grand
    "follow_caret": False,      # la zone suit la souris ; option : curseur de saisie
    "sound_alert": True,
    "toast_alert": True,
    "hide_when_away": False,
    "away_s": 8,
    "camera_index": 0,
    "theme": "sombre",
    "config_version": 2,
}

LENS_SIZES = {"petit": (420, 140), "moyen": (640, 210), "grand": (960, 320)}
LENS_ORDER = ["petit", "moyen", "grand"]
LENS_LABELS = {"petit": "Petit", "moyen": "Moyen", "grand": "Grand"}
HOTKEY_ID = 0x5A52          # identifiant du raccourci global Ctrl + Alt + Z
HOTKEY_PAUSE_ID = 0x5A50    # Ctrl + Alt + P : pause d'urgence, retire le voile
AWAY_RESET_S = 120          # voile d'absence : la détection est relancée toutes les 2 min

# ============================================================
# THEME
# ============================================================
# Palette fixe du voile et des notifications : ils s'affichent par-dessus
# n'importe quel contenu, ils gardent donc toujours le même rendu sombre.
BG = "#0F1012"
TEXT = "#EDEDEF"
MUTED = "#8B8D93"
BORDER = "#2A2C31"
ACCENT = "#3DDC97"
OK = "#3DDC97"
WARN = "#F2B544"
DANGER = "#FF6A5C"

THEMES = {
    "sombre": {
        "bg": "#0F1012", "side": "#121315", "surface": "#17181B", "surface2": "#1E2023",
        "hover": "#24262A", "seg_on": "#2E3035", "border": "#26282D", "text": "#EDEDEF",
        "muted": "#8B8D93", "faint": "#5C5F66", "accent": "#3DDC97", "warn": "#F2B544",
        "danger": "#FF6A5C", "primary": "#EDEDEF", "primary_text": "#0F1012",
        "primary_hover": "#FFFFFF", "track": "#33363C",
    },
    "clair": {
        "bg": "#F5F3EE", "side": "#EFECE5", "surface": "#FFFFFF", "surface2": "#F3F0EA",
        "hover": "#E9E5DC", "seg_on": "#FFFFFF", "border": "#E2DED5", "text": "#141517",
        "muted": "#6E7076", "faint": "#A3A5AA", "accent": "#12A06E", "warn": "#C98A12",
        "danger": "#D94A3D", "primary": "#141517", "primary_text": "#FFFFFF",
        "primary_hover": "#2A2C30", "track": "#D6D2C8",
    },
}
T = dict(THEMES["sombre"])

FONT_TEXT = "Segoe UI Variable Text"
FONT_DISPLAY = "Segoe UI Variable Display Semib"
FONT_NUM = "Bahnschrift"
FONT_NUM_LIGHT = "Bahnschrift Light"


def resolve_fonts():
    """Polices de Windows 11, avec repli sur Segoe UI si elles sont absentes."""
    global FONT_TEXT, FONT_DISPLAY, FONT_NUM, FONT_NUM_LIGHT
    from PyQt5.QtGui import QFontDatabase
    families = set(QFontDatabase().families())
    if FONT_TEXT not in families:
        FONT_TEXT = "Segoe UI"
    if FONT_DISPLAY not in families:
        FONT_DISPLAY = "Segoe UI Semibold"
    if FONT_NUM not in families:
        FONT_NUM = "Segoe UI"
    if FONT_NUM_LIGHT not in families:
        FONT_NUM_LIGHT = FONT_NUM


STYLE_TEMPLATE = """
* { font-family: '$ftext'; font-size: 10pt; color: $text; }
QMainWindow, QDialog { background: $bg; }
QWidget#page, QWidget#content { background: $bg; }
QFrame#side { background: $side; border: none; border-right: 1px solid $border; }
QFrame#card { background: $surface; border: 1px solid $border; border-radius: 14px; }
QFrame#hairline { background: $border; border: none; }
QLabel { background: transparent; }
QLabel#brand { font-family: '$fnum'; font-size: 12.5pt; font-weight: 600; letter-spacing: 3px; }
QLabel#brandSub { font-family: '$fnum'; color: $muted; font-size: 7.5pt; letter-spacing: 2px; }
QLabel#display { font-family: '$fdisp'; font-size: 22pt; }
QLabel#h1 { font-family: '$fdisp'; font-size: 20pt; }
QLabel#h2 { font-family: '$fdisp'; font-size: 13pt; }
QLabel#h3 { font-family: '$fdisp'; font-size: 11pt; }
QLabel#eyebrow { font-family: '$fnum'; color: $muted; font-size: 8pt; letter-spacing: 2px; }
QLabel#muted { color: $muted; }
QLabel#faint { color: $faint; font-size: 9pt; }
QLabel#metric { font-family: '$fnumlight'; font-size: 25pt; }
QLabel#value { font-family: '$fnum'; font-size: 10.5pt; }
QLabel#mono { font-family: '$fnum'; color: $muted; font-size: 9pt; }
QLabel#error { color: $danger; }
QPushButton {
    background: $surface2; border: 1px solid $border; border-radius: 9px;
    padding: 9px 18px; color: $text;
}
QPushButton:hover { background: $hover; }
QPushButton:disabled { color: $faint; background: $surface; }
QPushButton#primary { background: $primary; color: $primary_text; border: none; font-weight: 600; padding: 10px 22px; }
QPushButton#primary:hover { background: $primary_hover; }
QPushButton#primary:disabled { background: $surface2; color: $faint; }
QPushButton#ghost { background: transparent; border: none; color: $muted; padding: 6px 8px; }
QPushButton#ghost:hover { color: $text; }
QPushButton#dangerGhost { background: transparent; border: 1px solid $border; color: $danger; }
QPushButton#dangerGhost:hover { background: $hover; }
QPushButton#nav {
    text-align: left; background: transparent; border: none; border-radius: 9px;
    padding: 10px 12px; color: $muted;
}
QPushButton#nav:hover { color: $text; background: $hover; }
QPushButton#nav:checked { color: $text; background: $surface2; font-weight: 600; }
QFrame#segbox { background: $surface2; border: 1px solid $border; border-radius: 10px; }
QPushButton#seg { background: transparent; border: none; border-radius: 7px; padding: 6px 16px; color: $muted; }
QPushButton#seg:hover { color: $text; }
QPushButton#seg:checked { background: $seg_on; color: $text; font-weight: 600; }
QLineEdit {
    background: $surface2; border: 1px solid $border; border-radius: 9px;
    padding: 10px 12px; selection-background-color: $accent;
}
QLineEdit:focus { border-color: $muted; }
QSlider { background: transparent; }
QSlider::groove:horizontal { height: 4px; background: $track; border-radius: 2px; }
QSlider::sub-page:horizontal { background: $text; border-radius: 2px; }
QSlider::handle:horizontal {
    background: $surface; border: 2px solid $text; width: 12px; height: 12px;
    margin: -6px 0; border-radius: 8px;
}
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 4px 2px; }
QScrollBar::handle:vertical { background: $track; border-radius: 3px; min-height: 40px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QTableWidget { background: $surface; border: none; outline: 0; }
QTableWidget::item { padding: 0 10px; border-bottom: 1px solid $border; }
QHeaderView { background: $surface; }
QHeaderView::section {
    background: $surface; color: $muted; border: none; border-bottom: 1px solid $border;
    padding: 10px; font-family: '$fnum'; font-size: 8pt; letter-spacing: 1px;
}
QProgressBar { background: $surface2; border: none; border-radius: 2px; max-height: 4px; }
QProgressBar::chunk { background: $accent; border-radius: 2px; }
QMenu { background: $surface; border: 1px solid $border; padding: 6px; }
QMenu::item { padding: 7px 24px; border-radius: 6px; }
QMenu::item:selected { background: $hover; }
QMenu::separator { height: 1px; background: $border; margin: 4px 8px; }
QMessageBox { background: $surface; }
QMessageBox QLabel { color: $text; }
QToolTip { background: $surface2; color: $text; border: 1px solid $border; padding: 4px 8px; }
"""


def build_style(t):
    from string import Template
    return Template(STYLE_TEMPLATE).substitute(
        t, ftext=FONT_TEXT, fdisp=FONT_DISPLAY, fnum=FONT_NUM, fnumlight=FONT_NUM_LIGHT)


def apply_palette(app):
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(T["bg"]))
    pal.setColor(QPalette.WindowText, QColor(T["text"]))
    pal.setColor(QPalette.Base, QColor(T["surface"]))
    pal.setColor(QPalette.AlternateBase, QColor(T["surface2"]))
    pal.setColor(QPalette.Text, QColor(T["text"]))
    pal.setColor(QPalette.Button, QColor(T["surface2"]))
    pal.setColor(QPalette.ButtonText, QColor(T["text"]))
    pal.setColor(QPalette.Highlight, QColor(T["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor(T["primary_text"]))
    pal.setColor(QPalette.ToolTipBase, QColor(T["surface2"]))
    pal.setColor(QPalette.ToolTipText, QColor(T["text"]))
    pal.setColor(QPalette.PlaceholderText, QColor(T["faint"]))
    app.setPalette(pal)


def apply_theme(app, name):
    T.update(THEMES.get(name, THEMES["sombre"]))
    apply_palette(app)
    app.setStyleSheet(build_style(T))


def _shield_path(s, x=0.0, y=0.0):
    path = QPainterPath()
    path.moveTo(x + s * 0.50, y + s * 0.05)
    path.cubicTo(x + s * 0.66, y + s * 0.15, x + s * 0.80, y + s * 0.17, x + s * 0.89, y + s * 0.17)
    path.lineTo(x + s * 0.89, y + s * 0.48)
    path.cubicTo(x + s * 0.89, y + s * 0.74, x + s * 0.70, y + s * 0.88, x + s * 0.50, y + s * 0.96)
    path.cubicTo(x + s * 0.30, y + s * 0.88, x + s * 0.11, y + s * 0.74, x + s * 0.11, y + s * 0.48)
    path.lineTo(x + s * 0.11, y + s * 0.17)
    path.cubicTo(x + s * 0.20, y + s * 0.17, x + s * 0.34, y + s * 0.15, x + s * 0.50, y + s * 0.05)
    return path


def shield_pixmap(size=64, color=ACCENT, eye="#0B0C0E"):
    """Logo ZR-SHIELD : un bouclier plein contenant un oeil."""
    s = float(size)
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.fillPath(_shield_path(s), QColor(color))
    e = QPainterPath()
    e.moveTo(s * 0.27, s * 0.50)
    e.quadTo(s * 0.50, s * 0.28, s * 0.73, s * 0.50)
    e.quadTo(s * 0.50, s * 0.72, s * 0.27, s * 0.50)
    pen = QPen(QColor(eye), max(1.5, s * 0.055))
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawPath(e)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(eye))
    p.drawEllipse(QRectF(s * 0.43, s * 0.43, s * 0.14, s * 0.14))
    p.end()
    return pm


def format_distance(cm):
    if cm is None:
        return "Inconnue"
    if cm < 100:
        return f"{cm:.0f} cm"
    return f"{cm / 100:.1f} m".replace(".", ",")


# ============================================================
# COMPTES UTILISATEURS
# ============================================================
def _hash_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000).hex()


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


class UserStore:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.data = {"users": {}, "last_user": ""}
        if os.path.exists(USERS_FILE):
            try:
                with open(USERS_FILE, encoding="utf-8") as f:
                    self.data.update(json.load(f))
            except (OSError, ValueError):
                pass

    def _save(self):
        _write_json(USERS_FILE, self.data)

    def _key(self, name):
        for k in self.data["users"]:
            if k.lower() == name.lower():
                return k
        return None

    def has_users(self):
        return bool(self.data["users"])

    @property
    def last_user(self):
        return self.data.get("last_user", "")

    def create(self, name, password):
        if self._key(name):
            raise ValueError("Ce nom d'utilisateur est déjà utilisé.")
        salt = secrets.token_bytes(16)
        self.data["users"][name] = {
            "salt": salt.hex(),
            "hash": _hash_password(password, salt),
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        self.data["last_user"] = name
        self._save()
        return name

    def verify(self, name, password):
        key = self._key(name)
        if not key:
            return None
        rec = self.data["users"][key]
        if hmac.compare_digest(rec["hash"], _hash_password(password, bytes.fromhex(rec["salt"]))):
            self.data["last_user"] = key
            self._save()
            return key
        return None

    def change_password(self, name, old, new):
        if self.verify(name, old) is None:
            return False
        salt = secrets.token_bytes(16)
        self.data["users"][name].update(salt=salt.hex(), hash=_hash_password(new, salt))
        self._save()
        return True

    def created_on(self, name):
        rec = self.data["users"].get(name, {})
        try:
            return datetime.datetime.fromisoformat(rec["created"]).strftime("%d/%m/%Y")
        except (KeyError, ValueError):
            return ""


class UserProfile:
    """Réglages, profil facial et journal d'un utilisateur."""

    def __init__(self, name):
        self.name = name
        self.dir = os.path.join(DATA_DIR, "users", name)
        os.makedirs(self.dir, exist_ok=True)
        self.config_path = os.path.join(self.dir, "config.json")
        self.face_path = os.path.join(self.dir, "face.npy")
        self.journal_path = os.path.join(self.dir, "journal.txt")
        self.config = dict(DEFAULT_CONFIG)
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, encoding="utf-8") as f:
                    saved = json.load(f)
                self.config.update({k: v for k, v in saved.items() if k in DEFAULT_CONFIG})
                if saved.get("config_version", 1) < 2:
                    # Version 2 : la zone de lecture suit la souris par défaut.
                    self.config["follow_caret"] = False
                    self.config["config_version"] = 2
            except (OSError, ValueError):
                pass
        if self.config["lens_size"] not in LENS_SIZES:
            self.config["lens_size"] = "moyen"

    def save_config(self):
        _write_json(self.config_path, self.config)

    def load_face(self):
        if not os.path.exists(self.face_path):
            return None
        try:
            vec = np.load(self.face_path, allow_pickle=False).astype(np.float32).ravel()
        except (OSError, ValueError):
            return None
        return vec if vec.shape == (128,) else None

    def save_face(self, vec):
        np.save(self.face_path, np.asarray(vec, dtype=np.float32), allow_pickle=False)

    def clear_face(self):
        if os.path.exists(self.face_path):
            os.remove(self.face_path)
        self.config["ref_face_ratio"] = 0.0
        self.save_config()

    def log(self, event):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(f"{ts}\t{event}\n")

    def read_journal(self):
        rows = []
        if os.path.exists(self.journal_path):
            with open(self.journal_path, encoding="utf-8") as f:
                for line in f:
                    ts, _, event = line.rstrip("\n").partition("\t")
                    try:
                        dt = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        continue
                    rows.append((dt, event))
        return rows

    def clear_journal(self):
        if os.path.exists(self.journal_path):
            os.remove(self.journal_path)


# ============================================================
# DETECTION (thread caméra)
# ============================================================
def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class DetectionThread(QThread):
    analysis = pyqtSignal(dict)
    preview = pyqtSignal(QImage)
    enroll_progress = pyqtSignal(int)
    enroll_done = pyqtSignal(object, float)
    enroll_failed = pyqtSignal(str)
    failed = pyqtSignal(str)

    ENROLL_SAMPLES = 15
    ENROLL_TIMEOUT = 25.0
    OWNER_MEMORY = 3.0   # secondes pendant lesquelles on suit le visage du propriétaire

    def __init__(self, params, owner):
        super().__init__()
        self.params = dict(params)
        self.owner = owner
        self.paused = False
        self.preview_enabled = False
        self._running = True
        self._enroll_until = 0.0
        self._next_sample = 0.0
        self._samples = []
        self._ratios = []
        self._owner_box = None
        self._owner_time = 0.0

    def update_params(self, params):
        self.params = dict(params)

    def set_owner(self, vec):
        self.owner = vec

    def start_enroll(self):
        self._samples, self._ratios = [], []
        self._next_sample = 0.0
        self._enroll_until = time.time() + self.ENROLL_TIMEOUT

    def cancel_enroll(self):
        self._enroll_until = 0.0

    def stop(self):
        self._running = False

    def run(self):
        try:
            detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), 0.80, 0.3, 50)
            recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
        except cv2.error:
            self.failed.emit("Les modèles de reconnaissance sont introuvables dans le dossier models.")
            return

        backend = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(self.params.get("camera_index", 0)), backend)
        if not cap.isOpened():
            self.failed.emit("Impossible d'ouvrir la caméra. Vérifiez qu'elle est branchée "
                             "et qu'aucune autre application ne l'utilise.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        size = None
        misses = 0
        try:
            while self._running:
                if self.paused:
                    self.msleep(100)
                    continue
                ok, frame = cap.read()
                if not ok:
                    misses += 1
                    if misses > 60:
                        self.failed.emit("La caméra ne renvoie plus d'image.")
                        break
                    self.msleep(30)
                    continue
                misses = 0

                h, w = frame.shape[:2]
                if size != (w, h):
                    detector.setInputSize((w, h))
                    size = (w, h)
                _, raw = detector.detect(frame)
                faces = list(raw) if raw is not None else []
                now = time.time()

                if self._enroll_until:
                    self._enroll_step(frame, faces, recognizer, now, w)

                result = self._analyse(frame, faces, recognizer, now, w)
                self.analysis.emit(result)
                if self.preview_enabled:
                    self.preview.emit(self._draw(frame, result))
                self.msleep(10)
        finally:
            cap.release()

    @staticmethod
    def _feature(recognizer, frame, face):
        aligned = recognizer.alignCrop(frame, face)
        feat = recognizer.feature(aligned).ravel().astype(np.float32)
        n = np.linalg.norm(feat)
        return feat / n if n > 0 else feat

    def _enroll_step(self, frame, faces, recognizer, now, w):
        if now > self._enroll_until:
            self._enroll_until = 0.0
            self.enroll_failed.emit("Calibration interrompue. Placez-vous seul face à la caméra, "
                                    "le visage bien éclairé, puis recommencez.")
            return
        if len(faces) != 1 or now < self._next_sample or faces[0][14] < 0.9:
            return
        try:
            feat = self._feature(recognizer, frame, faces[0])
        except cv2.error:
            return
        self._samples.append(feat)
        self._ratios.append(float(faces[0][2]) / w)
        self._next_sample = now + 0.12
        self.enroll_progress.emit(int(100 * len(self._samples) / self.ENROLL_SAMPLES))
        if len(self._samples) >= self.ENROLL_SAMPLES:
            emb = np.mean(self._samples, axis=0)
            emb /= np.linalg.norm(emb)
            ratio = float(np.median(self._ratios))
            self._enroll_until = 0.0
            self.owner = emb.astype(np.float32)
            self.params["ref_face_ratio"] = ratio
            self.enroll_done.emit(self.owner, ratio)

    def _analyse(self, frame, faces, recognizer, now, w):
        p = self.params
        ref = p.get("ref_face_ratio") or 0.0
        max_cm = p["range_m"] * 100
        out = []
        for f in faces:
            box = (int(f[0]), int(f[1]), int(f[2]), int(f[3]))
            ratio = float(f[2]) / w
            dist = p["distance_cm"] * ref / ratio if ref > 0 and ratio > 0 else None
            if dist is not None and dist > max_cm:
                out.append({"box": box, "dist": dist, "score": None, "role": "ignored"})
                continue
            score = None
            if self.owner is not None:
                try:
                    score = float(self._feature(recognizer, frame, f) @ self.owner)
                except cv2.error:
                    score = None
            role = "owner" if score is not None and score >= p["match_threshold"] else "unknown"
            out.append({"box": box, "dist": dist, "score": score, "role": role})

        owners = [o for o in out if o["role"] == "owner"]
        if owners:
            best = max(owners, key=lambda o: o["score"])
            self._owner_box, self._owner_time = best["box"], now
        elif self._owner_box and now - self._owner_time < self.OWNER_MEMORY:
            # Tête tournée ou contre jour : on garde le visage qui occupe la place du propriétaire.
            for o in out:
                if o["role"] == "unknown" and _iou(o["box"], self._owner_box) > 0.35:
                    o["role"] = "owner"
                    self._owner_box = o["box"]
                    owners = [o]
                    break

        owner_dist = owners[0]["dist"] if owners else None
        return {
            "faces": out,
            "count": len(out),
            "intruders": sum(1 for o in out if o["role"] == "unknown"),
            "owner_present": bool(owners),
            "owner_dist": owner_dist,
            "frame_w": w,
        }

    @staticmethod
    def _draw(frame, result):
        img = frame.copy()
        colors = {"owner": (151, 220, 61), "unknown": (92, 106, 255), "ignored": (147, 141, 139)}
        for o in result["faces"]:
            x, y, bw, bh = o["box"]
            c = colors[o["role"]]
            cv2.rectangle(img, (x, y), (x + bw, y + bh), c, 2)
            if o["role"] == "owner":
                text = "Vous"
            elif o["role"] == "unknown":
                text = "Inconnu"
            else:
                text = "Hors zone"
            if o["dist"] is not None:
                text += "  " + format_distance(o["dist"]).replace("é", "e")
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(th + 8, y)
            cv2.rectangle(img, (x, ty - th - 8), (x + tw + 10, ty), c, -1)
            cv2.putText(img, text, (x + 5, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


# ============================================================
# SUIVI DE LA ZONE DE TRAVAIL (curseur de saisie / souris)
# ============================================================
if IS_WINDOWS:
    from ctypes import wintypes

    class _GUITHREADINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
            ("hwndActive", wintypes.HWND), ("hwndFocus", wintypes.HWND),
            ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
            ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND),
            ("rcCaret", wintypes.RECT),
        ]

    _user32 = ctypes.windll.user32
    _user32.GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.POINTER(_GUITHREADINFO)]
    _user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    _user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    _user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
    _user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]


def exclude_from_capture(hwnd):
    """Rend la fenêtre invisible aux captures d'écran (Windows 10 2004 et plus).
    On peut ainsi capturer l'écran réel sous le voile et le flouter en direct."""
    if not IS_WINDOWS:
        return False
    try:
        return bool(_user32.SetWindowDisplayAffinity(hwnd, 0x11))  # WDA_EXCLUDEFROMCAPTURE
    except Exception:
        return False


class FocusTracker:
    """Donne le point où l'utilisateur travaille : le curseur de saisie s'il vient
    de bouger, sinon la souris."""

    def __init__(self):
        self.mouse = None
        self.caret = None
        self.mouse_t = 0.0
        self.caret_t = 0.0

    @staticmethod
    def _caret_pos():
        if not IS_WINDOWS:
            return None
        info = _GUITHREADINFO()
        info.cbSize = ctypes.sizeof(info)
        if not _user32.GetGUIThreadInfo(0, ctypes.byref(info)) or not info.hwndCaret:
            return None
        r = info.rcCaret
        if r.right - r.left <= 0 and r.bottom - r.top <= 0:
            return None
        pt = wintypes.POINT(r.left, (r.top + r.bottom) // 2)
        _user32.ClientToScreen(info.hwndCaret, ctypes.byref(pt))
        return QPoint(pt.x, pt.y)

    def point(self, use_caret):
        now = time.time()
        m = QCursor.pos()
        if m != self.mouse:
            self.mouse, self.mouse_t = m, now
        if use_caret:
            c = self._caret_pos()
            if c is None:
                self.caret = None
            elif c != self.caret:
                self.caret, self.caret_t = c, now
            if self.caret is not None and self.caret_t >= self.mouse_t:
                return self.caret
        return self.mouse


# ============================================================
# VOILE DE PROTECTION
# ============================================================
class ScreenShield(QWidget):
    """Fenêtre transparente posée sur un écran. Elle affiche une version dégradée
    de l'écran, sauf dans la zone de lecture où le vrai contenu reste visible.
    Rien n'est peint dans cette zone : elle est donc parfaitement nette."""

    def __init__(self, screen):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.Tool | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.screen_ref = screen
        self.setGeometry(screen.geometry())
        self.frame_pm = None
        self.lens = None
        self.effect = "flou"
        self.mode = "intrus"
        self.alpha = 1.0
        self.capture_excluded = exclude_from_capture(int(self.winId()))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setOpacity(self.alpha)
        full = QRectF(self.rect())
        clip = QPainterPath()
        clip.addRect(full)
        if self.lens is not None:
            hole = QPainterPath()
            hole.addRoundedRect(QRectF(self.lens), 10, 10)
            clip = clip.subtracted(hole)
        p.setClipPath(clip)

        if self.effect == "voile" or self.frame_pm is None or self.mode == "absence":
            p.fillRect(self.rect(), QColor(8, 11, 16, 246))
        else:
            p.setRenderHint(QPainter.SmoothPixmapTransform, self.effect == "flou")
            p.drawPixmap(self.rect(), self.frame_pm)
            p.fillRect(self.rect(), QColor(8, 11, 16, 70))

        p.setRenderHint(QPainter.Antialiasing)
        if self.lens is not None:
            # Bordure tracée à l'extérieur de la zone de lecture, jamais dessus.
            p.setPen(QPen(QColor(43, 179, 255, 150), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(self.lens).adjusted(-1, -1, 1, 1), 11, 11)
        p.setClipping(False)

        text = "ZR-SHIELD  |  Contenu protégé" if self.mode == "intrus" else "ZR-SHIELD  |  Session masquée"
        p.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
        tw = p.fontMetrics().horizontalAdvance(text) + 28
        badge = QRectF((self.width() - tw) / 2, 14, tw, 30)
        if self.lens is None or not badge.intersects(QRectF(self.lens)):
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(18, 24, 33, 230))
            p.drawRoundedRect(badge, 15, 15)
            p.setPen(QColor(TEXT))
            p.drawText(badge, Qt.AlignCenter, text)

        if self.mode == "absence":
            p.setFont(QFont("Segoe UI", 16, QFont.DemiBold))
            p.setPen(QColor(MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "Écran masqué en votre absence")


class ShieldManager(QObject):
    FADE_MS = 1500

    def __init__(self):
        super().__init__()
        self.shields = []
        self.active = False
        self.fading = False
        self.mode = None
        self.cfg = dict(DEFAULT_CONFIG)
        self.live = True
        self.tracker = FocusTracker()
        self.capture_timer = QTimer(self, interval=150, timeout=self._capture)
        self.lens_timer = QTimer(self, interval=16, timeout=self._update_lens)
        self.fade = QVariantAnimation(self)
        self.fade.setStartValue(1.0)
        self.fade.setEndValue(0.0)
        self.fade.setDuration(self.FADE_MS)
        self.fade.setEasingCurve(QEasingCurve.InOutQuad)
        self.fade.valueChanged.connect(self._on_fade)
        self.fade.finished.connect(self._hide_all)
        app = QApplication.instance()
        app.screenAdded.connect(self._screens_changed)
        app.screenRemoved.connect(self._screens_changed)

    def _screens_changed(self, *_):
        was, mode = self.active and not self.fading, self.mode
        self.fade.stop()
        self._hide_all()
        for s in self.shields:
            s.deleteLater()
        self.shields = []
        if was:
            self.activate(mode)

    def _build(self):
        self.shields = [ScreenShield(s) for s in QApplication.screens()]
        self.live = all(s.capture_excluded for s in self.shields)

    def configure(self, cfg):
        self.cfg = dict(cfg)
        if self.active:
            for s in self.shields:
                s.effect = self.cfg["effect"]
            self._capture()
            self._update_lens(force=True)

    def activate(self, mode):
        if self.fading:
            # Une personne réapparaît pendant le fondu : protection immédiate.
            self.fade.stop()
            self.fading = False
            for s in self.shields:
                s.alpha = 1.0
                s.update()
        if self.active and mode == self.mode:
            return
        if not self.shields:
            self._build()
        self.mode = mode
        for s in self.shields:
            s.mode = mode
            s.effect = self.cfg["effect"]
            s.alpha = 1.0
        if not self.active:
            self._capture()     # avant l'affichage, pour ne jamais capturer le voile lui-même
            for s in self.shields:
                s.show()
                s.raise_()
            if self.live:
                self.capture_timer.start()
            self.lens_timer.start()
            self.active = True
        self._update_lens(force=True)

    def deactivate(self, animated=True):
        """Retire la protection, par défaut avec un fondu doux."""
        if not self.active:
            return
        if not animated:
            self.fade.stop()
            self._hide_all()
            return
        if not self.fading:
            self.fading = True
            self.fade.start()

    def _on_fade(self, value):
        for s in self.shields:
            s.alpha = float(value)
            s.update()

    def _hide_all(self):
        self.fading = False
        self.capture_timer.stop()
        self.lens_timer.stop()
        for s in self.shields:
            s.hide()
            s.frame_pm = None
            s.lens = None
            s.alpha = 1.0
        self.active = False
        self.mode = None

    def shutdown(self):
        self.deactivate(animated=False)
        for s in self.shields:
            s.close()
            s.deleteLater()
        self.shields = []

    def _capture(self):
        effect = self.cfg["effect"]
        for s in self.shields:
            if effect == "voile" or self.mode == "absence":
                s.frame_pm = None
            else:
                img = s.screen_ref.grabWindow(0).toImage()
                s.frame_pm = self._degrade(img, effect, int(self.cfg["strength"]))
            s.update()

    @staticmethod
    def _degrade(img, effect, strength):
        """Flou volontairement fort : même au réglage minimal, un texte de taille
        normale n'est plus lisible (réduction d'au moins 10 fois puis flou gaussien)."""
        w, h = img.width(), img.height()
        if w == 0 or h == 0:
            return None
        if effect == "pixels":
            block = 10 + strength * 3
            small = img.scaled(max(1, w // block), max(1, h // block),
                               Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            return QPixmap.fromImage(small)
        factor = 8 + strength * 2
        sw, sh = max(1, w // factor), max(1, h // factor)
        small = img.scaled(sw, sh, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        small = small.convertToFormat(QImage.Format_RGB32)
        ptr = small.constBits()
        ptr.setsize(small.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape(sh, small.bytesPerLine() // 4, 4)[:, :sw]
        arr = np.ascontiguousarray(cv2.GaussianBlur(arr, (0, 0), 1.6 + strength * 0.2))
        out = QImage(arr.data, sw, sh, arr.strides[0], QImage.Format_RGB32).copy()
        return QPixmap.fromImage(out)

    def _update_lens(self, force=False):
        lens_on = self.mode == "intrus" and self.cfg["lens_enabled"]
        pt = self.tracker.point(self.cfg["follow_caret"]) if lens_on else None
        lw, lh = LENS_SIZES.get(self.cfg["lens_size"], LENS_SIZES["moyen"])
        for s in self.shields:
            lens = None
            geo = s.geometry()
            if pt is not None and geo.contains(pt):
                r = QRect(0, 0, min(lw, geo.width()), min(lh, geo.height()))
                r.moveCenter(pt - geo.topLeft())
                r.moveLeft(max(0, min(r.left(), geo.width() - r.width())))
                r.moveTop(max(0, min(r.top(), geo.height() - r.height())))
                lens = r
            if force or lens != s.lens:
                old = s.lens
                s.lens = lens
                if force or old is None or lens is None:
                    s.update()
                else:
                    s.update(old.united(lens).adjusted(-6, -6, 6, 6))
                    s.update(0, 0, s.width(), 50)   # le badge peut être masqué ou révélé


# ============================================================
# ALERTE DISCRETE : NOTIFICATION ET SON
# ============================================================
class Toast(QWidget):
    """Petite notification en bas à droite de l'écran. Elle ne prend jamais le
    focus et laisse passer les clics : le travail n'est pas interrompu."""

    W, H = 400, 78

    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.Tool | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.resize(self.W, self.H)
        self.title = ""
        self.body = ""
        self.color = QColor(DANGER)
        self.alpha = 1.0
        self.icon = shield_pixmap(28)
        self._hide_timer = QTimer(self, singleShot=True, timeout=self._fade_out)
        self._fade = QVariantAnimation(self)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.setDuration(600)
        self._fade.valueChanged.connect(self._set_alpha)
        self._fade.finished.connect(self.hide)
        exclude_from_capture(int(self.winId()))

    def notify(self, title, body, color=DANGER, duration=4000):
        self.title, self.body, self.color = title, body, QColor(color)
        self._fade.stop()
        self.alpha = 1.0
        screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        geo = screen.availableGeometry()
        self.move(geo.right() - self.W - 20, geo.bottom() - self.H - 20)
        self.show()
        self.raise_()
        self.update()
        self._hide_timer.start(duration)

    def _fade_out(self):
        self._fade.start()

    def _set_alpha(self, v):
        self.alpha = float(v)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setOpacity(self.alpha)
        p.setRenderHint(QPainter.Antialiasing)
        card = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setPen(QPen(QColor(BORDER), 1))
        p.setBrush(QColor(18, 24, 33, 245))
        p.drawRoundedRect(card, 10, 10)
        p.setPen(Qt.NoPen)
        p.setBrush(self.color)
        p.drawRoundedRect(QRectF(1, 14, 4, self.H - 28), 2, 2)
        p.drawPixmap(18, (self.H - 28) // 2, self.icon)
        p.setPen(QColor(TEXT))
        p.setFont(QFont("Segoe UI", 10, QFont.DemiBold))
        p.drawText(QRectF(58, 14, self.W - 72, 24), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        p.setPen(QColor(MUTED))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(QRectF(58, 38, self.W - 72, 24), Qt.AlignLeft | Qt.AlignVCenter, self.body)


class AlertSound:
    """Carillon court et doux, généré une seule fois puis joué en arrière-plan."""

    def __init__(self):
        self.path = os.path.join(DATA_DIR, "alerte.wav")
        if not os.path.exists(self.path):
            try:
                self._generate()
            except OSError:
                self.path = None

    def _generate(self):
        import wave
        rate = 44100
        chunks = []
        for freq, dur in ((880.0, 0.11), (1318.5, 0.16)):
            t = np.arange(int(rate * dur)) / rate
            env = np.minimum(1.0, t / 0.008) * np.exp(-t * 14)
            chunks.append(0.28 * env * np.sin(2 * np.pi * freq * t))
        data = (np.concatenate(chunks) * 32767).astype(np.int16)
        os.makedirs(DATA_DIR, exist_ok=True)
        with wave.open(self.path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(data.tobytes())

    def play(self):
        if IS_WINDOWS and self.path:
            import winsound
            winsound.PlaySound(self.path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        else:
            QApplication.beep()


# ============================================================
# COMPOSANTS D'INTERFACE
# ============================================================
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
        "septembre", "octobre", "novembre", "décembre"]


def make_label(text="", name=None, wrap=False):
    lbl = QLabel(text)
    if name:
        lbl.setObjectName(name)
    lbl.setWordWrap(wrap)
    return lbl


def eyebrow(text):
    return make_label(text.upper(), "eyebrow")


def make_button(text, name=None):
    b = QPushButton(text)
    if name:
        b.setObjectName(name)
    b.setCursor(Qt.PointingHandCursor)
    return b


def hairline():
    f = QFrame()
    f.setObjectName("hairline")
    f.setFixedHeight(1)
    return f


def make_card(margins=(24, 22, 24, 22), spacing=12):
    frame = QFrame()
    frame.setObjectName("card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return frame, lay


def make_page(kicker, title, subtitle=None):
    """Page défilante avec en-tête : surtitre, titre, sous-titre."""
    page = QWidget()
    page.setObjectName("page")
    outer = QVBoxLayout(page)
    outer.setContentsMargins(0, 0, 0, 0)
    content = QWidget()
    content.setObjectName("content")
    lay = QVBoxLayout(content)
    lay.setContentsMargins(44, 36, 44, 36)
    lay.setSpacing(18)
    head = QVBoxLayout()
    head.setSpacing(6)
    head.addWidget(eyebrow(kicker))
    head.addWidget(make_label(title, "h1"))
    if subtitle:
        head.addWidget(make_label(subtitle, "muted", wrap=True))
    lay.addLayout(head)
    lay.addSpacing(6)
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    area.setWidget(content)
    outer.addWidget(area)
    return page, lay


def draw_icon(name, color, size=18):
    """Icônes au trait, dessinées pour rester nettes à toutes les tailles."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color), 1.6)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    s = float(size)
    if name == "overview":
        for x, y in ((0.12, 0.12), (0.56, 0.12), (0.12, 0.56), (0.56, 0.56)):
            p.drawRoundedRect(QRectF(s * x, s * y, s * 0.32, s * 0.32), 2.5, 2.5)
    elif name == "camera":
        p.drawRoundedRect(QRectF(s * 0.08, s * 0.26, s * 0.84, s * 0.56), 3, 3)
        p.drawEllipse(QRectF(s * 0.35, s * 0.37, s * 0.30, s * 0.30))
        p.drawLine(QPoint(int(s * 0.36), int(s * 0.26)), QPoint(int(s * 0.42), int(s * 0.14)))
        p.drawLine(QPoint(int(s * 0.42), int(s * 0.14)), QPoint(int(s * 0.58), int(s * 0.14)))
        p.drawLine(QPoint(int(s * 0.58), int(s * 0.14)), QPoint(int(s * 0.64), int(s * 0.26)))
    elif name == "settings":
        for y, k in ((0.25, 0.65), (0.50, 0.35), (0.75, 0.58)):
            p.drawLine(QPoint(int(s * 0.1), int(s * y)), QPoint(int(s * 0.9), int(s * y)))
            p.setBrush(QColor(color))
            p.drawEllipse(QRectF(s * k - 2.6, s * y - 2.6, 5.2, 5.2))
            p.setBrush(Qt.NoBrush)
    elif name == "history":
        p.drawEllipse(QRectF(s * 0.1, s * 0.1, s * 0.8, s * 0.8))
        p.drawLine(QPoint(int(s * 0.5), int(s * 0.5)), QPoint(int(s * 0.5), int(s * 0.27)))
        p.drawLine(QPoint(int(s * 0.5), int(s * 0.5)), QPoint(int(s * 0.67), int(s * 0.6)))
    elif name == "account":
        p.drawEllipse(QRectF(s * 0.33, s * 0.1, s * 0.34, s * 0.34))
        path = QPainterPath()
        path.moveTo(s * 0.13, s * 0.9)
        path.cubicTo(s * 0.15, s * 0.58, s * 0.85, s * 0.58, s * 0.87, s * 0.9)
        p.drawPath(path)
    elif name == "about":
        p.drawEllipse(QRectF(s * 0.1, s * 0.1, s * 0.8, s * 0.8))
        p.drawLine(QPoint(int(s * 0.5), int(s * 0.46)), QPoint(int(s * 0.5), int(s * 0.7)))
        p.setBrush(QColor(color))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QRectF(s * 0.5 - 1.3, s * 0.3 - 1.3, 2.6, 2.6))
    elif name == "logout":
        p.drawLine(QPoint(int(s * 0.42), int(s * 0.5)), QPoint(int(s * 0.9), int(s * 0.5)))
        p.drawLine(QPoint(int(s * 0.75), int(s * 0.35)), QPoint(int(s * 0.9), int(s * 0.5)))
        p.drawLine(QPoint(int(s * 0.75), int(s * 0.65)), QPoint(int(s * 0.9), int(s * 0.5)))
        path = QPainterPath()
        path.moveTo(s * 0.55, s * 0.18)
        path.lineTo(s * 0.12, s * 0.18)
        path.lineTo(s * 0.12, s * 0.82)
        path.lineTo(s * 0.55, s * 0.82)
        p.drawPath(path)
    p.end()
    return pm


def dot_pixmap(color, size=10):
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    p.drawEllipse(QRectF(1, 1, size - 2, size - 2))
    p.end()
    return pm


class Switch(QAbstractButton):
    """Interrupteur animé."""

    def __init__(self, checked=False):
        super().__init__()
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(42, 24)
        self._pos = 1.0 if checked else 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._move)
        self.toggled.connect(self._animate)

    def _animate(self, on):
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def _move(self, v):
        self._pos = float(v)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        off, on = QColor(T["track"]), QColor(T["accent"])
        k = self._pos
        track = QColor(int(off.red() + (on.red() - off.red()) * k),
                       int(off.green() + (on.green() - off.green()) * k),
                       int(off.blue() + (on.blue() - off.blue()) * k))
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 2, 42, 20), 10, 10)
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(QRectF(3 + 18 * k, 5, 14, 14))


class Segmented(QFrame):
    """Sélecteur segmenté : une seule option active."""
    changed = pyqtSignal(str)

    def __init__(self, options, current):
        super().__init__()
        self.setObjectName("segbox")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self.group = QButtonGroup(self)
        self.buttons = {}
        for key, text in options:
            b = make_button(text, "seg")
            b.setCheckable(True)
            b.setChecked(key == current)
            bold = QFont(FONT_TEXT, 10)
            bold.setWeight(QFont.DemiBold)
            b.setMinimumWidth(QFontMetrics(bold).horizontalAdvance(text) + 38)
            b.clicked.connect(lambda _, k=key: self.changed.emit(k))
            self.group.addButton(b)
            self.buttons[key] = b
            lay.addWidget(b)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def set(self, key):
        if key in self.buttons:
            self.buttons[key].setChecked(True)


class Avatar(QWidget):
    def __init__(self, name, size=36):
        super().__init__()
        self.letter = (name[:1] or "?").upper()
        self.setFixedSize(size, size)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(T["accent"]))
        p.drawEllipse(self.rect())
        p.setPen(QColor("#0B0C0E"))
        f = QFont(FONT_NUM)
        f.setPixelSize(int(self.height() * 0.46))
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.drawText(self.rect(), Qt.AlignCenter, self.letter)


class StatusOrb(QWidget):
    """Indicateur d'état : un bouclier entouré d'ondes qui pulsent quand la
    protection travaille."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(150, 150)
        self.state = "idle"
        self.phase = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(2600)
        self._anim.setLoopCount(-1)
        self._anim.valueChanged.connect(self._tick)

    def _tick(self, v):
        self.phase = float(v)
        self.update()

    def color(self):
        return QColor({"active": T["accent"], "alert": T["danger"], "pause": T["warn"],
                       "away": T["warn"], "calib": T["text"]}.get(self.state, T["faint"]))

    def set_state(self, state):
        if state == self.state:
            return
        self.state = state
        if state in ("active", "alert", "calib"):
            self._anim.setDuration(1300 if state == "alert" else 2600)
            if self._anim.state() != QVariantAnimation.Running:
                self._anim.start()
        else:
            self._anim.stop()
            self.phase = 0.0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        col = self.color()
        animated = self._anim.state() == QVariantAnimation.Running
        for i in range(3):
            if animated:
                t = (self.phase + i / 3.0) % 1.0
                r = 34 + t * 38
                a = int(110 * (1 - t))
            else:
                r = 44 + i * 13
                a = 38 - i * 10
            ring = QColor(col)
            ring.setAlpha(max(0, a))
            p.setPen(QPen(ring, 1.4))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c, r, r)
        fill = QColor(col)
        fill.setAlpha(34)
        p.setPen(QPen(col, 1.6))
        p.setBrush(fill)
        p.drawEllipse(c, 33, 33)
        pm = shield_pixmap(34, col.name())
        p.drawPixmap(int(c.x() - 17), int(c.y() - 17), pm)


class FieldView(QWidget):
    """Vue de dessus de l'espace devant l'écran. Chaque visage détecté est placé
    selon sa distance estimée et sa position dans l'image."""

    FOV = 68.0

    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 220)
        self.data = None
        self.range_cm = 300.0
        self.phase = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(1400)
        self._anim.setLoopCount(-1)
        self._anim.valueChanged.connect(self._tick)
        self._anim.start()

    def _tick(self, v):
        self.phase = float(v)
        if self.data and self.data.get("intruders"):
            self.update()

    def set_data(self, data, range_m):
        self.data = data
        self.range_cm = range_m * 100
        self.update()

    def _r(self, cm, R):
        return R * min(1.0, math.sqrt(max(cm, 0) / self.range_cm))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        origin = QPointF(w / 2, h - 26)
        R = min(w / 2 - 16, h - 46)
        live = self.data is not None
        faint, muted, text = QColor(T["faint"]), QColor(T["muted"]), QColor(T["text"])
        accent = QColor(T["accent"])

        # Cône de vision de la caméra
        half = self.FOV / 2
        cone = QPainterPath()
        cone.moveTo(origin)
        cone.arcTo(QRectF(origin.x() - R, origin.y() - R, 2 * R, 2 * R), 90 - half, self.FOV)
        cone.closeSubpath()
        fill = QColor(accent if live else faint)
        fill.setAlpha(22 if live else 12)
        p.setPen(Qt.NoPen)
        p.setBrush(fill)
        p.drawPath(cone)

        # Anneaux de distance
        p.setFont(QFont(FONT_NUM, 7))
        rings = [c for c in (50, 100, 200, 300, 400, 500) if c <= self.range_cm]
        for cm in rings:
            r = self._r(cm, R)
            col = QColor(faint)
            col.setAlpha(120)
            pen = QPen(col, 1, Qt.DashLine if cm != rings[-1] else Qt.SolidLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawArc(QRectF(origin.x() - r, origin.y() - r, 2 * r, 2 * r), 16 * 15, 16 * 150)
            p.setPen(faint)
            label = f"{cm} cm" if cm < 100 else f"{cm // 100} m"
            ang = math.radians(90 - half - 6)
            p.drawText(QPointF(origin.x() + r * math.cos(ang) + 4, origin.y() - r * math.sin(ang) + 3), label)

        # Ecran
        p.setPen(Qt.NoPen)
        p.setBrush(text if live else faint)
        p.drawRoundedRect(QRectF(origin.x() - 26, origin.y() + 2, 52, 5), 2, 2)
        p.setPen(muted)
        p.setFont(QFont(FONT_NUM, 7))
        p.drawText(QRectF(origin.x() - 40, origin.y() + 9, 80, 14), Qt.AlignCenter, "ÉCRAN")

        if not live:
            p.setPen(muted)
            p.setFont(QFont(FONT_TEXT, 9))
            p.drawText(QRectF(0, h * 0.32, w, 20), Qt.AlignCenter, "Caméra en veille")
            return

        fw = max(1, self.data.get("frame_w", 640))
        for f in self.data["faces"]:
            x, y, bw, bh = f["box"]
            nx = (x + bw / 2) / fw
            theta = math.radians((nx - 0.5) * self.FOV)
            dist = f["dist"] if f["dist"] is not None else self.range_cm * 0.6
            r = max(18.0, self._r(dist, R))
            pt = QPointF(origin.x() + r * math.sin(theta), origin.y() - r * math.cos(theta))
            role = f["role"]
            col = QColor(T["accent"] if role == "owner" else T["danger"] if role == "unknown" else T["faint"])
            if role == "unknown":
                halo = QColor(col)
                halo.setAlpha(int(90 * (1 - self.phase)))
                p.setPen(Qt.NoPen)
                p.setBrush(halo)
                p.drawEllipse(pt, 7 + 12 * self.phase, 7 + 12 * self.phase)
            p.setPen(QPen(QColor(T["surface"]), 2))
            p.setBrush(col if role != "ignored" else Qt.NoBrush)
            if role == "ignored":
                p.setPen(QPen(col, 1.5))
            p.drawEllipse(pt, 6, 6)
            p.setPen(text if role != "ignored" else faint)
            p.setFont(QFont(FONT_NUM, 8))
            name = "Vous" if role == "owner" else "Inconnu" if role == "unknown" else "Hors zone"
            p.drawText(QPointF(pt.x() + 10, pt.y() + 4), name)


class MetricTile(QFrame):
    def __init__(self, label, value="0", caption=""):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(4)
        lay.addWidget(eyebrow(label))
        self.value = make_label(value, "metric")
        lay.addWidget(self.value)
        self.caption = make_label(caption, "faint")
        lay.addWidget(self.caption)

    def set(self, text, caption=None):
        if self.value.text() != text:
            self.value.setText(text)
        if caption is not None and self.caption.text() != caption:
            self.caption.setText(caption)


class SettingsGroup(QFrame):
    """Carte de réglages : des lignes séparées par un filet."""

    def __init__(self, title, subtitle=None):
        super().__init__()
        self.setObjectName("card")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(24, 20, 24, 8)
        self.lay.setSpacing(0)
        self.lay.addWidget(make_label(title, "h3"))
        if subtitle:
            self.lay.addSpacing(4)
            self.lay.addWidget(make_label(subtitle, "muted", wrap=True))
        self.lay.addSpacing(10)
        self._rows = 0

    def add_row(self, title, desc, control):
        if self._rows:
            self.lay.addWidget(hairline())
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 14, 0, 14)
        col = QVBoxLayout()
        col.setSpacing(3)
        col.addWidget(make_label(title))
        lbl = None
        if desc:
            lbl = make_label(desc, "faint", wrap=True)
            col.addWidget(lbl)
        h.addLayout(col, 1)
        h.addSpacing(28)
        h.addWidget(control, 0, Qt.AlignVCenter)
        self.lay.addWidget(row)
        self._rows += 1
        return lbl


# ============================================================
# CONNEXION
# ============================================================
class BrandPanel(QWidget):
    """Panneau gauche de l'écran de connexion."""

    def __init__(self):
        super().__init__()
        self.setFixedWidth(400)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#0B0C0E"))
        # Motif : ondes de détection qui partent du bas
        origin = QPointF(w * 0.5, h + 40)
        for i in range(9):
            r = 120 + i * 62
            col = QColor(ACCENT)
            col.setAlpha(max(8, 70 - i * 7))
            p.setPen(QPen(col, 1))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(origin, r, r)
        p.drawPixmap(44, 44, shield_pixmap(38, ACCENT))
        p.setPen(QColor(TEXT))
        f = QFont(FONT_NUM, 12)
        f.setWeight(QFont.DemiBold)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 3)
        p.setFont(f)
        p.drawText(QPointF(94, 64), "ZR-SHIELD")
        f2 = QFont(FONT_NUM, 7)
        f2.setLetterSpacing(QFont.AbsoluteSpacing, 2)
        p.setFont(f2)
        p.setPen(QColor(MUTED))
        p.drawText(QPointF(95, 80), "BMM SYSTEMS")

        p.setPen(QColor(TEXT))
        p.setFont(QFont(FONT_DISPLAY, 24))
        p.setFont(QFont(FONT_DISPLAY, 22))
        p.drawText(QRectF(44, 136, w - 88, 140), Qt.AlignLeft,
                   "Votre écran.\nVos yeux\nseulement.")
        p.setPen(QColor(MUTED))
        p.setFont(QFont(FONT_TEXT, 10))
        p.drawText(QRectF(44, 296, w - 88, 90), Qt.AlignLeft | Qt.TextWordWrap,
                   "Dès qu'une personne regarde votre écran derrière vous, "
                   "le contenu se brouille pour elle. Vous continuez à travailler.")
        p.setFont(QFont(FONT_NUM, 7))
        p.setPen(QColor(MUTED))
        p.drawText(QPointF(44, h - 36), f"VERSION {APP_VERSION}   |   TRAITEMENT 100 % LOCAL")


class LoginDialog(QDialog):
    def __init__(self, store):
        super().__init__()
        self.store = store
        self.username = None
        self.setWindowTitle(APP_NAME)
        self.setFixedSize(860, 560)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(BrandPanel())

        form = QWidget()
        f = QVBoxLayout(form)
        f.setContentsMargins(64, 0, 64, 0)
        f.setSpacing(0)
        f.addStretch(3)
        self.kicker = eyebrow("")
        f.addWidget(self.kicker)
        f.addSpacing(8)
        self.title = make_label("", "h1")
        f.addWidget(self.title)
        f.addSpacing(6)
        self.subtitle = make_label("", "muted", wrap=True)
        f.addWidget(self.subtitle)
        f.addSpacing(28)

        self.ed_user = QLineEdit()
        self.ed_user.setPlaceholderText("Nom d'utilisateur")
        self.ed_pass = QLineEdit()
        self.ed_pass.setPlaceholderText("Mot de passe")
        self.ed_pass.setEchoMode(QLineEdit.Password)
        self.ed_conf = QLineEdit()
        self.ed_conf.setPlaceholderText("Confirmer le mot de passe")
        self.ed_conf.setEchoMode(QLineEdit.Password)
        for ed in (self.ed_user, self.ed_pass, self.ed_conf):
            f.addWidget(ed)
            f.addSpacing(10)

        self.hint = make_label("3 à 24 caractères : lettres, chiffres, point ou tiret bas (_).", "faint", wrap=True)
        f.addWidget(self.hint)
        self.error = make_label("", "error", wrap=True)
        f.addWidget(self.error)
        f.addSpacing(16)

        self.btn_ok = make_button("", "primary")
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self._submit)
        f.addWidget(self.btn_ok)
        f.addSpacing(14)

        row = QHBoxLayout()
        self.switch_text = make_label("", "muted")
        self.btn_switch = make_button("", "ghost")
        self.btn_switch.setStyleSheet(f"color:{T['text']}; font-weight:600; padding:4px 2px;")
        self.btn_switch.clicked.connect(lambda: self._set_mode(not self.create_mode))
        row.addWidget(self.switch_text)
        row.addSpacing(6)
        row.addWidget(self.btn_switch)
        row.addStretch()
        f.addLayout(row)
        f.addStretch(4)
        root.addWidget(form, 1)

        self._set_mode(not store.has_users())
        if store.last_user and not self.create_mode:
            self.ed_user.setText(store.last_user)
            self.ed_pass.setFocus()

    def _set_mode(self, create):
        self.create_mode = create
        self.error.clear()
        self.ed_conf.setVisible(create)
        self.hint.setVisible(create)
        if create:
            self.kicker.setText("NOUVEAU COMPTE")
            self.title.setText("Créer votre espace")
            self.subtitle.setText("Chaque utilisateur dispose de son propre profil facial, "
                                  "de ses réglages et de son historique.")
            self.btn_ok.setText("Créer mon compte")
            self.switch_text.setText("Déjà inscrit ?")
            self.btn_switch.setText("Se connecter")
        else:
            self.kicker.setText("CONNEXION")
            self.title.setText("Bon retour")
            self.subtitle.setText("Connectez-vous pour activer la protection de votre écran.")
            self.btn_ok.setText("Se connecter")
            self.switch_text.setText("Nouvel utilisateur ?")
            self.btn_switch.setText("Créer un compte")

    def _submit(self):
        name = self.ed_user.text().strip()
        pw = self.ed_pass.text()
        if self.create_mode:
            if not USERNAME_RE.match(name):
                self.error.setText("Nom d'utilisateur invalide.")
                return
            if len(pw) < 6:
                self.error.setText("Le mot de passe doit contenir au moins 6 caractères.")
                return
            if pw != self.ed_conf.text():
                self.error.setText("Les mots de passe ne correspondent pas.")
                return
            try:
                self.username = self.store.create(name, pw)
            except ValueError as e:
                self.error.setText(str(e))
                return
        else:
            self.username = self.store.verify(name, pw)
            if not self.username:
                self.error.setText("Nom d'utilisateur ou mot de passe incorrect.")
                self.ed_pass.selectAll()
                self.ed_pass.setFocus()
                return
        self.accept()


# ============================================================
# FENETRE PRINCIPALE
# ============================================================
NAV_ITEMS = [
    ("overview", "Vue d'ensemble"),
    ("camera", "Caméra"),
    ("settings", "Réglages"),
    ("history", "Historique"),
    ("account", "Mon compte"),
    ("about", "À propos"),
]


class MainWindow(QMainWindow):
    def __init__(self, profile, controller):
        super().__init__()
        self.profile = profile
        self.controller = controller
        self.cfg = profile.config
        self.owner = profile.load_face()
        if self.owner is None:
            self.cfg["ref_face_ratio"] = 0.0

        self.thread = None
        self.state = "idle"         # idle | running | paused
        self.calibrating = False
        self.testing = False
        self.alert = False
        self.away = False
        self.alerts = 0
        self.last_result = None
        self._intr_since = None
        self._intr_last = 0.0
        self._alert_start = 0.0
        self._owner_last = time.time()
        self._away_reset = 0.0
        self._protect_start = None
        self._protected_s = 0.0
        self._status_key = None
        self._quitting = False
        self._tray_hint_shown = False

        self.shields = ShieldManager()
        self.shields.configure(self.cfg)
        self.toast = Toast()
        self.sound = AlertSound()

        self._save_timer = QTimer(self, singleShot=True, interval=400, timeout=self.profile.save_config)
        self._test_timer = QTimer(self, singleShot=True, interval=6000, timeout=self._end_test)
        self._clock = QTimer(self, interval=1000, timeout=self._tick)
        self._clock.start()

        self.setWindowTitle(f"{APP_NAME} | {profile.name}")
        self.setMinimumSize(1180, 760)
        self.resize(1300, 840)
        self._build()
        self._build_tray()
        self._refresh_status()
        self._refresh_profile()
        self._refresh_activity()
        self._hotkey = self._register_hotkey()
        if not self._hotkey:
            self.lbl_hotkey.setText("Le raccourci Ctrl + Alt + Z est déjà pris par une autre application.")
        self._hotkey_pause = self._register_hotkey(HOTKEY_PAUSE_ID, "P")

    # ---------- construction ----------
    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self._build_sidebar())
        self.pages = QStackedWidget()
        for page in (self._page_overview(), self._page_camera(), self._page_settings(),
                     self._page_history(), self._page_account(), self._page_about()):
            self.pages.addWidget(page)
        h.addWidget(self.pages, 1)
        self._go(0)

    def _build_sidebar(self):
        side = QFrame()
        side.setObjectName("side")
        side.setFixedWidth(248)
        v = QVBoxLayout(side)
        v.setContentsMargins(18, 26, 18, 20)
        v.setSpacing(2)

        brand = QHBoxLayout()
        brand.setSpacing(12)
        logo = QLabel()
        logo.setPixmap(shield_pixmap(32, ACCENT))
        brand.addWidget(logo)
        col = QVBoxLayout()
        col.setSpacing(0)
        col.addWidget(make_label(APP_NAME, "brand"))
        col.addWidget(make_label("BMM SYSTEMS", "brandSub"))
        brand.addLayout(col)
        brand.addStretch()
        v.addLayout(brand)
        v.addSpacing(34)
        v.addWidget(eyebrow("Menu"))
        v.addSpacing(8)

        self.nav_group = QButtonGroup(self)
        self.nav_buttons = []
        for i, (key, text) in enumerate(NAV_ITEMS):
            b = make_button("   " + text, "nav")
            b.setCheckable(True)
            b.setIconSize(QSize(18, 18))
            b.clicked.connect(lambda _, idx=i: self._go(idx))
            self.nav_group.addButton(b)
            self.nav_buttons.append((key, b))
            v.addWidget(b)
        v.addStretch()

        # Etat compact
        self.side_state = make_label("", "mono")
        v.addWidget(self.side_state)
        v.addSpacing(14)
        v.addWidget(hairline())
        v.addSpacing(14)

        user = QHBoxLayout()
        user.setSpacing(12)
        user.addWidget(Avatar(self.profile.name, 36))
        col = QVBoxLayout()
        col.setSpacing(0)
        col.addWidget(make_label(self.profile.name, "h3"))
        col.addWidget(make_label("Session ouverte", "faint"))
        user.addLayout(col, 1)
        self.btn_logout = make_button("", "ghost")
        self.btn_logout.setToolTip("Se déconnecter")
        self.btn_logout.setIconSize(QSize(18, 18))
        self.btn_logout.clicked.connect(self._logout)
        user.addWidget(self.btn_logout)
        v.addLayout(user)
        self._paint_nav_icons()
        return side

    def _paint_nav_icons(self):
        for key, b in self.nav_buttons:
            icon = QIcon()
            icon.addPixmap(draw_icon(key, T["muted"]), QIcon.Normal, QIcon.Off)
            icon.addPixmap(draw_icon(key, T["text"]), QIcon.Normal, QIcon.On)
            icon.addPixmap(draw_icon(key, T["text"]), QIcon.Active, QIcon.Off)
            b.setIcon(icon)
        if hasattr(self, "btn_logout"):
            self.btn_logout.setIcon(QIcon(draw_icon("logout", T["muted"])))

    def _page_overview(self):
        now = datetime.datetime.now()
        page, lay = make_page(f"{JOURS[now.weekday()]} {now.day} {MOIS[now.month - 1]}",
                              f"Bonjour, {self.profile.name}")

        # Bloc principal : état + champ de vision
        top = QHBoxLayout()
        top.setSpacing(18)
        hero, hl = make_card((28, 26, 28, 26), 0)
        row = QHBoxLayout()
        row.setSpacing(26)
        self.orb = StatusOrb()
        row.addWidget(self.orb, 0, Qt.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(6)
        self.lbl_kicker = eyebrow("")
        self.lbl_title = make_label("", "display")
        self.lbl_desc = make_label("", "muted", wrap=True)
        col.addWidget(self.lbl_kicker)
        col.addWidget(self.lbl_title)
        col.addWidget(self.lbl_desc)
        col.addSpacing(18)
        btns = QHBoxLayout()
        btns.setSpacing(10)
        self.btn_main = make_button("", "primary")
        self.btn_main.clicked.connect(self._toggle_protection)
        self.btn_pause = make_button("Pause")
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_test = make_button("Tester le rendu", "ghost")
        self.btn_test.clicked.connect(self._test_effect)
        btns.addWidget(self.btn_main)
        btns.addWidget(self.btn_pause)
        btns.addWidget(self.btn_test)
        btns.addStretch()
        col.addLayout(btns)
        col.addStretch()
        row.addLayout(col, 1)
        hl.addLayout(row)
        top.addWidget(hero, 3)

        field, fl = make_card((22, 20, 22, 14), 6)
        fh = QHBoxLayout()
        fh.addWidget(eyebrow("Champ de vision"))
        fh.addStretch()
        self.lbl_field = make_label("", "mono")
        fh.addWidget(self.lbl_field)
        fl.addLayout(fh)
        self.field = FieldView()
        fl.addWidget(self.field, 1)
        top.addWidget(field, 2)
        lay.addLayout(top)

        # Indicateurs
        tiles = QHBoxLayout()
        tiles.setSpacing(18)
        self.tile_faces = MetricTile("Visages", "0", "dans le champ de la caméra")
        self.tile_alerts = MetricTile("Alertes", "0", "depuis l'activation")
        self.tile_dist = MetricTile("Votre distance", "...", "")
        self.tile_time = MetricTile("Temps protégé", "0 min", "durant cette session")
        for t in (self.tile_faces, self.tile_alerts, self.tile_dist, self.tile_time):
            tiles.addWidget(t)
        lay.addLayout(tiles)

        # Profil facial + activité récente
        bottom = QHBoxLayout()
        bottom.setSpacing(18)
        cal, cl = make_card((24, 22, 24, 22), 10)
        ch = QHBoxLayout()
        ch.addWidget(make_label("Profil facial", "h2"))
        ch.addStretch()
        self.lbl_profile_pill = QLabel()
        ch.addWidget(self.lbl_profile_pill)
        cl.addLayout(ch)
        self.lbl_profile = make_label("", "muted", wrap=True)
        cl.addWidget(self.lbl_profile)
        cl.addSpacing(6)
        drow = QHBoxLayout()
        drow.addWidget(make_label("Distance habituelle"))
        drow.addStretch()
        self.vl_dist = make_label(f"{self.cfg['distance_cm']} cm", "value")
        drow.addWidget(self.vl_dist)
        cl.addLayout(drow)
        self.sl_dist = QSlider(Qt.Horizontal)
        self.sl_dist.setRange(25, 120)
        self.sl_dist.setValue(int(self.cfg["distance_cm"]))
        self.sl_dist.valueChanged.connect(self._on_distance)
        cl.addWidget(self.sl_dist)
        cl.addSpacing(6)
        self.cal_progress = QProgressBar()
        self.cal_progress.setTextVisible(False)
        self.cal_progress.setRange(0, 100)
        self.cal_progress.hide()
        cl.addWidget(self.cal_progress)
        crow = QHBoxLayout()
        self.btn_cal = make_button("Calibrer mon visage")
        self.btn_cal.clicked.connect(self._calibrate)
        self.btn_cal_cancel = make_button("Annuler", "ghost")
        self.btn_cal_cancel.clicked.connect(lambda: self._end_calibration(None))
        self.btn_cal_cancel.hide()
        crow.addWidget(self.btn_cal)
        crow.addWidget(self.btn_cal_cancel)
        crow.addStretch()
        cl.addLayout(crow)
        cl.addStretch()
        bottom.addWidget(cal, 1)

        act, al = make_card((24, 22, 24, 16), 0)
        ah = QHBoxLayout()
        ah.addWidget(make_label("Activité récente", "h2"))
        ah.addStretch()
        more = make_button("Tout voir", "ghost")
        more.clicked.connect(lambda: self._go(3))
        ah.addWidget(more)
        al.addLayout(ah)
        al.addSpacing(8)
        self.activity_box = QVBoxLayout()
        self.activity_box.setSpacing(0)
        al.addLayout(self.activity_box)
        al.addStretch()
        bottom.addWidget(act, 1)
        lay.addLayout(bottom)
        lay.addStretch()
        return page

    def _page_camera(self):
        page, lay = make_page("Caméra", "Ce que voit ZR-SHIELD",
                              "Aperçu en direct. Le traitement est entièrement local : "
                              "aucune image n'est enregistrée ni envoyée.")
        row = QHBoxLayout()
        row.setSpacing(18)
        view, vl = make_card((10, 10, 10, 10), 0)
        self.cam_view = QLabel("La caméra s'active pendant la protection ou la calibration.")
        self.cam_view.setObjectName("muted")
        self.cam_view.setAlignment(Qt.AlignCenter)
        self.cam_view.setMinimumSize(560, 420)
        self.cam_view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.cam_view.setStyleSheet(f"background:{T['bg']}; border-radius:10px;")
        vl.addWidget(self.cam_view)
        row.addWidget(view, 3)

        side, sl = make_card((24, 22, 24, 22), 14)
        sl.addWidget(eyebrow("Lecture en direct"))
        self.cam_stats = {}
        for key, text in (("count", "Visages détectés"), ("owner", "Vous"), ("unknown", "Inconnus"),
                          ("dist", "Votre distance")):
            r = QHBoxLayout()
            r.addWidget(make_label(text, "muted"))
            r.addStretch()
            val = make_label("...", "value")
            r.addWidget(val)
            self.cam_stats[key] = val
            sl.addLayout(r)
        sl.addSpacing(8)
        sl.addWidget(hairline())
        sl.addSpacing(8)
        sl.addWidget(eyebrow("Légende"))
        for color, text in ((T["accent"], "Vous"), (T["danger"], "Personne inconnue"),
                            (T["faint"], "Hors de la zone surveillée")):
            r = QHBoxLayout()
            d = QLabel()
            d.setPixmap(dot_pixmap(color))
            r.addWidget(d)
            r.addWidget(make_label(text, "muted"))
            r.addStretch()
            sl.addLayout(r)
        sl.addStretch()
        row.addWidget(side, 1)
        lay.addLayout(row, 1)
        return page

    def _slider_control(self, key, lo, hi, fmt, scale=1.0, step=1):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(14)
        s = QSlider(Qt.Horizontal)
        s.setFixedWidth(220)
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setPageStep(step)
        s.setValue(int(round(self.cfg[key] * scale)))
        val = make_label(fmt(self.cfg[key]), "value")
        val.setFixedWidth(64)
        val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        def changed(x):
            real = x / scale if scale != 1.0 else x
            val.setText(fmt(real))
            self._set_cfg(key, real)

        s.valueChanged.connect(changed)
        h.addWidget(s)
        h.addWidget(val)
        return w

    def _switch(self, key):
        sw = Switch(bool(self.cfg[key]))
        sw.toggled.connect(lambda on: self._set_cfg(key, on))
        return sw

    def _page_settings(self):
        page, lay = make_page("Préférences", "Réglages",
                              "Chaque modification est appliquée et enregistrée immédiatement.")

        g = SettingsGroup("Apparence")
        theme = Segmented([("sombre", "Sombre"), ("clair", "Clair")], self.cfg["theme"])
        theme.changed.connect(self._set_theme)
        g.add_row("Thème", "Choisissez l'apparence de l'application.", theme)
        lay.addWidget(g)

        g = SettingsGroup("Rendu de protection", "Ce que voient les personnes autour de vous pendant une alerte.")
        eff = Segmented([("flou", "Flou"), ("pixels", "Pixels"), ("voile", "Voile")], self.cfg["effect"])
        eff.changed.connect(lambda k: self._set_cfg("effect", k))
        g.add_row("Effet", "Le flou reste le plus discret, le voile le plus opaque.", eff)
        g.add_row("Intensité", "Même au minimum, le texte n'est plus lisible.",
                  self._slider_control("strength", 1, 10, lambda v: f"{int(v)} / 10"))
        lay.addWidget(g)

        g = SettingsGroup("Zone de lecture", "Une fenêtre parfaitement nette suit votre souris en temps réel.")
        g.add_row("Zone nette", "Désactivez-la pour flouter la totalité de l'écran.", self._switch("lens_enabled"))
        g.add_row("Curseur de saisie", "Suivre aussi le curseur quand vous écrivez au clavier.",
                  self._switch("follow_caret"))
        self.seg_lens = Segmented([(k, LENS_LABELS[k]) for k in LENS_ORDER], self.cfg["lens_size"])
        self.seg_lens.changed.connect(self._set_lens_size)
        self.lbl_hotkey = g.add_row("Taille", "Raccourci Ctrl + Alt + Z, même en arrière-plan.", self.seg_lens)
        lay.addWidget(g)

        g = SettingsGroup("Alerte discrète", "Vous êtes prévenu sans être interrompu.")
        g.add_row("Son", "Deux notes brèves à l'arrivée d'une personne.", self._switch("sound_alert"))
        g.add_row("Notification", "Un message en bas à droite de l'écran, qui disparaît seul.",
                  self._switch("toast_alert"))
        lay.addWidget(g)

        g = SettingsGroup("Détection")
        g.add_row("Portée", "Au-delà, les visages sont ignorés.",
                  self._slider_control("range_m", 10, 50, lambda v: format_distance(v * 100), scale=10))
        g.add_row("Exigence de reconnaissance", "Plus elle est haute, moins un tiers peut être confondu avec vous.",
                  self._slider_control("match_threshold", 25, 60, lambda v: f"{v * 100:.0f} %", scale=100))
        g.add_row("Délai avant protection", "Temps de présence avant le flou.",
                  self._slider_control("trigger_ms", 0, 2000, lambda v: f"{int(v)} ms", step=50))
        g.add_row("Délai avant retour", "Suivi d'un fondu de 1,5 s.",
                  self._slider_control("release_ms", 300, 5000, lambda v: f"{int(v)} ms", step=100))
        lay.addWidget(g)

        g = SettingsGroup("Absence")
        g.add_row("Masquer en mon absence", "Tout l'écran est voilé quand vous quittez votre poste.",
                  self._switch("hide_when_away"))
        g.add_row("Délai d'absence", None,
                  self._slider_control("away_s", 3, 60, lambda v: f"{int(v)} s"))
        lay.addWidget(g)

        g = SettingsGroup("Caméra")
        cam = Segmented([("0", "Caméra 1"), ("1", "Caméra 2"), ("2", "Caméra 3")], str(self.cfg["camera_index"]))
        cam.changed.connect(lambda k: self._set_cfg("camera_index", int(k)))
        g.add_row("Source vidéo", "Prise en compte au prochain démarrage de la protection.", cam)
        lay.addWidget(g)
        lay.addStretch()
        return page

    def _page_history(self):
        page, lay = make_page("Journal", "Historique", "Les événements de protection de votre compte.")
        card, cl = make_card((8, 8, 8, 8), 0)
        bar = QHBoxLayout()
        bar.setContentsMargins(16, 10, 10, 6)
        self.lbl_count = make_label("", "mono")
        bar.addWidget(self.lbl_count)
        bar.addStretch()
        b_ref = make_button("Actualiser", "ghost")
        b_exp = make_button("Exporter en CSV", "ghost")
        b_clr = make_button("Effacer", "ghost")
        b_clr.setStyleSheet(f"color:{T['danger']};")
        b_ref.clicked.connect(self._load_history)
        b_exp.clicked.connect(self._export_history)
        b_clr.clicked.connect(self._clear_history)
        for b in (b_ref, b_exp, b_clr):
            bar.addWidget(b)
        cl.addLayout(bar)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["DATE", "HEURE", "ÉVÉNEMENT"])
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(44)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setIconSize(QSize(10, 10))
        hh = self.table.horizontalHeader()
        hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 150)
        self.table.setColumnWidth(1, 120)
        self.table.setMinimumHeight(480)
        cl.addWidget(self.table)
        lay.addWidget(card, 1)
        return page

    def _page_account(self):
        page, lay = make_page("Compte", "Mon compte")

        head, hl = make_card((26, 24, 26, 24), 0)
        row = QHBoxLayout()
        row.setSpacing(18)
        row.addWidget(Avatar(self.profile.name, 60))
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(make_label(self.profile.name, "h1"))
        col.addWidget(make_label(f"Membre depuis le {self.controller.store.created_on(self.profile.name)}", "muted"))
        row.addLayout(col, 1)
        out = make_button("Se déconnecter")
        out.clicked.connect(self._logout)
        row.addWidget(out, 0, Qt.AlignVCenter)
        hl.addLayout(row)
        lay.addWidget(head)

        sec, sl = make_card((26, 22, 26, 24), 10)
        sl.addWidget(make_label("Mot de passe", "h2"))
        sl.addWidget(make_label("Au moins 6 caractères. Il est stocké sous forme chiffrée (PBKDF2).", "muted"))
        sl.addSpacing(6)
        self.pw_old, self.pw_new, self.pw_conf = QLineEdit(), QLineEdit(), QLineEdit()
        for ed, ph in ((self.pw_old, "Mot de passe actuel"), (self.pw_new, "Nouveau mot de passe"),
                       (self.pw_conf, "Confirmer le nouveau mot de passe")):
            ed.setEchoMode(QLineEdit.Password)
            ed.setPlaceholderText(ph)
            ed.setMaximumWidth(420)
            sl.addWidget(ed)
        self.pw_msg = make_label("", wrap=True)
        sl.addWidget(self.pw_msg)
        r = QHBoxLayout()
        b = make_button("Mettre à jour", "primary")
        b.clicked.connect(self._change_password)
        r.addWidget(b)
        r.addStretch()
        sl.addLayout(r)
        lay.addWidget(sec)

        face, fl = make_card((26, 22, 26, 24), 10)
        fl.addWidget(make_label("Données biométriques", "h2"))
        fl.addWidget(make_label("Votre profil facial est une empreinte numérique de 128 valeurs. Il ne contient "
                                "aucune photo et ne quitte jamais cet ordinateur.", "muted", wrap=True))
        r = QHBoxLayout()
        b = make_button("Supprimer mon profil facial", "dangerGhost")
        b.clicked.connect(self._delete_face)
        r.addWidget(b)
        r.addStretch()
        fl.addLayout(r)
        lay.addWidget(face)
        lay.addStretch()
        return page

    def _page_about(self):
        page, lay = make_page("À propos", f"{APP_NAME} {APP_VERSION}",
                              "Protection visuelle intelligente de l'écran.")

        how, hl = make_card((26, 24, 26, 24), 16)
        hl.addWidget(make_label("Comment ça marche", "h2"))
        steps = (
            ("01", "Calibration", "ZR-SHIELD apprend votre visage et votre distance de travail habituelle."),
            ("02", "Surveillance", "La caméra observe l'espace devant l'écran, en local, plusieurs fois par seconde."),
            ("03", "Protection", "Dès qu'une autre personne regarde, l'écran se brouille pour elle. "
                                 "Une zone nette suit votre souris pour que vous continuiez à travailler."),
        )
        grid = QHBoxLayout()
        grid.setSpacing(28)
        for num, title, text in steps:
            c = QVBoxLayout()
            c.setSpacing(6)
            n = make_label(num, "metric")
            n.setStyleSheet(f"color:{T['accent']};")
            c.addWidget(n)
            c.addWidget(make_label(title, "h3"))
            c.addWidget(make_label(text, "muted", wrap=True))
            c.addStretch()
            grid.addLayout(c, 1)
        hl.addLayout(grid)
        lay.addWidget(how)

        lim, ll = make_card((26, 22, 26, 24), 10)
        ll.addWidget(make_label("Limites connues", "h2"))
        for t in (
            "La zone nette reste lisible pour une personne qui regarde exactement au même endroit. "
            "Réduisez sa taille ou désactivez-la pour une confidentialité maximale.",
            "Un téléphone qui filme l'écran ou une personne hors du champ de la caméra ne sont pas détectés.",
            "Un court délai sépare l'arrivée d'une personne et l'activation de la protection.",
            "Pour une protection physique complète, associez ZR-SHIELD à un filtre de confidentialité.",
        ):
            r = QHBoxLayout()
            r.setSpacing(12)
            d = QLabel()
            d.setPixmap(dot_pixmap(T["faint"], 8))
            r.addWidget(d, 0, Qt.AlignTop)
            r.addWidget(make_label(t, "muted", wrap=True), 1)
            ll.addLayout(r)
        lay.addWidget(lim)

        cred, crl = make_card((26, 22, 26, 24), 4)
        crl.addWidget(eyebrow("Conception"))
        crl.addSpacing(6)
        crl.addWidget(make_label("Bella Marie Mathilde", "h2"))
        crl.addWidget(make_label("Master 2 Cybersécurité et Blockchain, Keyce Informatique, Yaoundé", "muted"))
        crl.addWidget(make_label("Projet ZYRHA OS, BMM Systems", "faint"))
        lay.addWidget(cred)
        lay.addStretch()
        return page

    def _build_tray(self):
        self.tray = QSystemTrayIcon(QIcon(shield_pixmap(64, ACCENT)), self)
        menu = QMenu()
        menu.addAction("Ouvrir ZR-SHIELD", self._show_window)
        self.act_toggle = menu.addAction("Activer la protection", self._toggle_protection)
        menu.addSeparator()
        menu.addAction("Quitter", self._quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda reason: reason == QSystemTrayIcon.Trigger and self._show_window())
        self.tray.show()

    def _go(self, index):
        self.pages.setCurrentIndex(index)
        self.nav_buttons[index][1].setChecked(True)
        if self.thread:
            self.thread.preview_enabled = index == 1
        if index == 3:
            self._load_history()

    # ---------- thème ----------
    def _set_theme(self, name):
        self.cfg["theme"] = name
        self._save_timer.start()
        apply_theme(QApplication.instance(), name)
        self._paint_nav_icons()
        self.cam_view.setStyleSheet(f"background:{T['bg']}; border-radius:10px;")
        self._status_key = None
        self._refresh_status()
        self._refresh_profile()
        self._refresh_activity()
        for w in self.findChildren(QWidget):
            w.update()

    # ---------- thread caméra ----------
    def _ensure_thread(self):
        if self.thread is not None:
            return
        t = DetectionThread(self.cfg, self.owner)
        t.analysis.connect(self._on_analysis)
        t.preview.connect(self._on_preview)
        t.enroll_progress.connect(self.cal_progress.setValue)
        t.enroll_done.connect(self._on_enrolled)
        t.enroll_failed.connect(lambda msg: self._end_calibration(msg))
        t.failed.connect(self._on_thread_failed)
        t.preview_enabled = self.pages.currentIndex() == 1
        self.thread = t
        t.start()

    def _release_thread(self):
        if self.state != "idle" or self.calibrating or self.thread is None:
            return
        t, self.thread = self.thread, None
        t.stop()
        t.wait(3000)
        self.last_result = None
        self.field.set_data(None, self.cfg["range_m"])
        self.cam_view.clear()
        self.cam_view.setText("La caméra s'active pendant la protection ou la calibration.")

    def _on_thread_failed(self, msg):
        if self.sender() is not self.thread:
            return
        if self.state == "running" and self.away:
            # Caméra indisponible pendant l'absence : le voile reste, nouvel essai à la prochaine relance.
            self._log(f"Caméra indisponible : {msg}")
            self._keep_thread(self.thread)
            self.thread = None
            return
        t, self.thread = self.thread, None
        if t:
            t.stop()
            t.wait(1000)
        self.calibrating = False
        self._reset_detection()
        self._stop_clock()
        self.state = "idle"
        self.cal_progress.hide()
        self.btn_cal_cancel.hide()
        self.btn_cal.setEnabled(True)
        self.field.set_data(None, self.cfg["range_m"])
        self._apply_mode()
        self._refresh_status()
        QMessageBox.warning(self, "Caméra", msg)

    # ---------- actions ----------
    def _log(self, event):
        self.profile.log(event)
        self._refresh_activity()

    def _toggle_protection(self):
        if self.state == "idle":
            self._start()
        else:
            self._stop()

    def _start(self):
        if self.owner is None:
            QMessageBox.information(self, "Calibration requise",
                                    "Calibrez d'abord votre visage depuis la vue d'ensemble.")
            return
        self.state = "running"
        self.alerts = 0
        self._protected_s = 0.0
        self._protect_start = time.time()
        self._reset_detection()
        self._ensure_thread()
        self.thread.paused = False
        self._log("Protection activée")
        self._refresh_status()

    def _toggle_pause(self):
        if self.state == "running":
            self.state = "paused"
            if self.thread:
                self.thread.paused = True
            self._reset_detection()
            self._stop_clock()
            self._log("Protection mise en pause")
        elif self.state == "paused":
            self.state = "running"
            self._protect_start = time.time()
            self._ensure_thread()
            self.thread.paused = False
            self._log("Protection reprise")
        self._apply_mode()
        self._refresh_status()

    def _stop(self):
        if self.state == "idle":
            return
        self.state = "idle"
        self._reset_detection()
        self._stop_clock()
        self._apply_mode()
        self._release_thread()
        self._log("Protection désactivée")
        self._refresh_status()

    def _stop_clock(self):
        if self._protect_start is not None:
            self._protected_s += time.time() - self._protect_start
            self._protect_start = None

    def _tick(self):
        self._check_away_reset()
        total = self._protected_s + (time.time() - self._protect_start if self._protect_start else 0)
        m = int(total // 60)
        self.tile_time.set(f"{m // 60} h {m % 60:02d}" if m >= 60 else f"{m} min")

    def _test_effect(self):
        self.testing = True
        self._test_timer.start()
        self._apply_mode()
        self._notify_intrusion(1)

    def _end_test(self):
        self.testing = False
        self._apply_mode()

    def _on_distance(self, v):
        self.vl_dist.setText(f"{v} cm")
        self._set_cfg("distance_cm", v)
        self._refresh_profile()

    def _calibrate(self):
        if self.calibrating:
            return
        if self.thread and self.thread.paused:
            self.thread.paused = False
        self.calibrating = True
        self._reset_detection()
        self._apply_mode()
        self._ensure_thread()
        self.thread.start_enroll()
        self.cal_progress.setValue(0)
        self.cal_progress.show()
        self.btn_cal.setEnabled(False)
        self.btn_cal_cancel.show()
        self._refresh_status()

    def _on_enrolled(self, emb, ratio):
        self.owner = emb
        self.cfg["ref_face_ratio"] = ratio
        self.profile.save_face(emb)
        self.profile.save_config()
        self._log(f"Calibration réalisée à {self.cfg['distance_cm']} cm")
        self._end_calibration(None, success=True)

    def _end_calibration(self, error, success=False):
        if not self.calibrating:
            return
        self.calibrating = False
        if self.thread:
            self.thread.cancel_enroll()
            self.thread.update_params(self.cfg)
            self.thread.set_owner(self.owner)
            if self.state == "paused":
                self.thread.paused = True
        self.cal_progress.hide()
        self.btn_cal.setEnabled(True)
        self.btn_cal_cancel.hide()
        self._reset_detection()
        self._release_thread()
        self._refresh_profile()
        self._refresh_status()
        if error:
            QMessageBox.warning(self, "Calibration", error)
        elif success:
            self.toast.notify("Calibration terminée", "Votre visage et votre distance sont enregistrés.",
                              ACCENT, 3000)

    # ---------- logique de protection ----------
    def _reset_detection(self):
        if self.alert:
            self._log(f"Fin d'alerte après {time.time() - self._alert_start:.0f} s")
        self.alert = False
        self.away = False
        self._intr_since = None
        self._owner_last = time.time()

    def _check_away_reset(self):
        """Voile d'absence : toutes les 2 min, la caméra et la détection sont relancées
        (caméra figée après une veille, par exemple). Le voile reste en place jusqu'à
        ce que le propriétaire soit reconnu."""
        if self.state != "running" or self.calibrating or not self.away:
            return
        now = time.time()
        if now - self._away_reset < AWAY_RESET_S:
            return
        self._away_reset = now
        old, self.thread = self.thread, None
        if old:
            for sig in (old.analysis, old.preview, old.failed):
                try:
                    sig.disconnect()
                except TypeError:
                    pass
            old.stop()
            old.wait(3000)
            self._keep_thread(old)
        self._ensure_thread()
        self.thread.paused = False
        self._log("Absence prolongée, détection relancée")

    def _keep_thread(self, t):
        """Un thread bloqué dans la lecture caméra ne doit pas être détruit en cours d'exécution."""
        t.stop()
        self._old_threads = [o for o in getattr(self, "_old_threads", []) if o.isRunning()]
        if t.isRunning():
            self._old_threads.append(t)

    def _on_analysis(self, r):
        if self.sender() is not self.thread:
            return
        self.last_result = r
        self.field.set_data(r, self.cfg["range_m"])
        unknown = r["intruders"]
        self.lbl_field.setText(f"{r['count']} VISAGE{'S' if r['count'] > 1 else ''}")
        self.tile_faces.set(str(r["count"]),
                            "dont 1 inconnu" if unknown == 1 else
                            f"dont {unknown} inconnus" if unknown > 1 else "dans le champ de la caméra")
        dist = format_distance(r["owner_dist"]) if r["owner_present"] else "..."
        self.tile_dist.set(dist, "estimée en direct" if r["owner_present"] else "vous n'êtes pas détecté")
        if self.pages.currentIndex() == 1:
            self.cam_stats["count"].setText(str(r["count"]))
            self.cam_stats["owner"].setText("Détecté" if r["owner_present"] else "Absent")
            self.cam_stats["unknown"].setText(str(unknown))
            self.cam_stats["dist"].setText(dist)
        if self.calibrating or self.state != "running":
            return

        now = time.time()
        new_alert = 0
        if unknown > 0:
            if self._intr_since is None:
                self._intr_since = now
            self._intr_last = now
            if not self.alert and (now - self._intr_since) * 1000 >= self.cfg["trigger_ms"]:
                self.alert = True
                self._alert_start = now
                self.alerts += 1
                new_alert = unknown
                self._log(f"Regard indiscret détecté : {unknown} personne{'s' if unknown > 1 else ''}")
        else:
            self._intr_since = None
            if self.alert and (now - self._intr_last) * 1000 >= self.cfg["release_ms"]:
                self.alert = False
                self._log(f"Fin d'alerte après {now - self._alert_start:.0f} s")

        if r["owner_present"]:
            self._owner_last = now
        away = self.cfg["hide_when_away"] and now - self._owner_last >= self.cfg["away_s"]
        if away != self.away:
            self.away = away
            self._away_reset = now
            self._log("Absence détectée, écran masqué" if away else "Retour à l'écran")

        self.tile_alerts.set(str(self.alerts))
        self._apply_mode()
        if new_alert:
            self._notify_intrusion(new_alert)
        self._refresh_status()

    def _notify_intrusion(self, count):
        if self.cfg["sound_alert"]:
            self.sound.play()
        if self.cfg["toast_alert"]:
            title = "Une personne regarde votre écran" if count == 1 else f"{count} personnes regardent votre écran"
            self.toast.notify(title, "Écran flouté, votre zone de lecture reste nette.")

    def _set_lens_size(self, size):
        self._set_cfg("lens_size", size)
        self.seg_lens.set(size)

    def _cycle_lens_size(self):
        i = (LENS_ORDER.index(self.cfg["lens_size"]) + 1) % len(LENS_ORDER)
        self._set_lens_size(LENS_ORDER[i])
        self.toast.notify(f"Zone de lecture : {LENS_LABELS[LENS_ORDER[i]]}",
                          "Ctrl + Alt + Z pour changer de taille.", ACCENT, 1800)

    def _register_hotkey(self, hotkey_id=HOTKEY_ID, key="Z"):
        if not IS_WINDOWS:
            return False
        # MOD_ALT | MOD_CONTROL | MOD_NOREPEAT
        return bool(_user32.RegisterHotKey(int(self.winId()), hotkey_id, 0x0001 | 0x0002 | 0x4000, ord(key)))

    def _emergency_pause(self):
        if self.state == "running" and not self.calibrating:
            self._toggle_pause()
            self.toast.notify("Protection en pause", "Ctrl + Alt + P pour la reprendre.", WARN, 3000)
        elif self.state == "paused":
            self._toggle_pause()
            self.toast.notify("Protection reprise", "Ctrl + Alt + P pour la mettre en pause.", ACCENT, 2000)

    def nativeEvent(self, event_type, message):
        if IS_WINDOWS and event_type == b"windows_generic_MSG":
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == 0x0312 and msg.wParam == HOTKEY_ID:     # WM_HOTKEY
                self._cycle_lens_size()
                return True, 0
            if msg.message == 0x0312 and msg.wParam == HOTKEY_PAUSE_ID:
                self._emergency_pause()
                return True, 0
        return super().nativeEvent(event_type, message)

    def _apply_mode(self):
        running = self.state == "running" and not self.calibrating
        if self.testing or (running and self.alert):
            self.shields.activate("intrus")
        elif running and self.away:
            self.shields.activate("absence")
        else:
            self.shields.deactivate()

    def _set_cfg(self, key, value):
        if key in ("strength", "trigger_ms", "release_ms", "away_s", "camera_index", "distance_cm"):
            value = int(value)
        self.cfg[key] = value
        self._save_timer.start()
        self.shields.configure(self.cfg)
        if self.thread:
            self.thread.update_params(self.cfg)
        if key == "range_m":
            self.field.set_data(self.last_result if self.thread else None, value)

    # ---------- affichage ----------
    def _refresh_status(self):
        if self.calibrating:
            key = ("calib", "Calibration", "Restez immobile",
                   "Seul face à la caméra, à votre distance habituelle. Quelques secondes suffisent.")
        elif self.state == "idle":
            key = ("idle", "Protection inactive", "Écran exposé",
                   "Activez la protection pour surveiller votre environnement."
                   if self.owner is not None else
                   "Commencez par calibrer votre visage, ci-dessous.")
        elif self.state == "paused":
            key = ("pause", "En pause", "Surveillance suspendue", "La caméra est en veille. Reprenez quand vous le souhaitez.")
        elif self.alert:
            key = ("alert", "Alerte", "Regard indiscret",
                   "Votre écran est brouillé pour les personnes autour de vous. Votre zone de lecture reste nette.")
        elif self.away:
            key = ("away", "Absence", "Écran masqué", "Vous avez quitté votre poste. L'écran reste voilé jusqu'à votre retour.")
        else:
            key = ("active", "Protection active", "Vous êtes seul",
                   "Personne d'autre ne regarde votre écran.")
        if key == self._status_key:
            return
        self._status_key = key
        state, kicker, title, desc = key
        self.orb.set_state(state)
        color = self.orb.color().name()
        self.lbl_kicker.setText(kicker.upper())
        self.lbl_kicker.setStyleSheet(f"color:{color};")
        self.lbl_title.setText(title)
        self.lbl_desc.setText(desc)
        self.side_state.setText(f"●  {kicker.upper()}")
        self.side_state.setStyleSheet(f"color:{color};")

        idle = self.state == "idle"
        self.btn_main.setText("Activer la protection" if idle else "Désactiver")
        self.btn_main.setEnabled(not self.calibrating)
        self.btn_pause.setVisible(not idle)
        self.btn_pause.setEnabled(not self.calibrating)
        self.btn_pause.setText("Reprendre" if self.state == "paused" else "Pause")
        self.act_toggle.setText("Activer la protection" if idle else "Désactiver la protection")
        self.tray.setToolTip(f"{APP_NAME} | {kicker}")

    def _refresh_profile(self):
        if self.owner is not None:
            pill, color = "ENREGISTRÉ", T["accent"]
            self.lbl_profile.setText(f"Calibré à {self.cfg['distance_cm']} cm. Recalibrez si votre "
                                     "installation ou l'éclairage changent.")
            self.btn_cal.setText("Recalibrer")
        else:
            pill, color = "À FAIRE", T["warn"]
            self.lbl_profile.setText("Installez-vous à votre distance habituelle, seul face à la caméra. "
                                     "La calibration dure environ cinq secondes.")
            self.btn_cal.setText("Calibrer mon visage")
        self.lbl_profile_pill.setText(pill)
        self.lbl_profile_pill.setStyleSheet(
            f"color:{color}; border:1px solid {color}; border-radius:10px; padding:2px 10px;"
            f"font-family:'{FONT_NUM}'; font-size:7.5pt; letter-spacing:1px;")

    def _event_color(self, event):
        if event.startswith("Regard indiscret"):
            return T["danger"]
        if event.startswith(("Absence", "Protection mise en pause")):
            return T["warn"]
        if event.startswith(("Protection activée", "Protection démarrée", "Calibration", "Protection reprise")):
            return T["accent"]
        return T["faint"]

    def _refresh_activity(self):
        while self.activity_box.count():
            item = self.activity_box.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        rows = self.profile.read_journal()[-5:][::-1]
        if not rows:
            self.activity_box.addWidget(make_label("Aucun événement pour le moment.", "faint"))
            return
        today = datetime.date.today()
        for i, (dt, event) in enumerate(rows):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 9, 0, 9)
            h.setSpacing(12)
            d = QLabel()
            d.setPixmap(dot_pixmap(self._event_color(event), 8))
            h.addWidget(d)
            h.addWidget(make_label(event), 1)
            when = dt.strftime("%H:%M") if dt.date() == today else dt.strftime("%d/%m")
            h.addWidget(make_label(when, "mono"))
            if i:
                self.activity_box.addWidget(hairline())
            self.activity_box.addWidget(w)

    def _on_preview(self, img):
        if self.pages.currentIndex() != 1:
            return
        target = self.cam_view.size()
        pm = QPixmap.fromImage(img).scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        rounded = QPixmap(pm.size())
        rounded.fill(Qt.transparent)
        p = QPainter(rounded)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(rounded.rect()), 10, 10)
        p.setClipPath(path)
        p.drawPixmap(0, 0, pm)
        p.end()
        self.cam_view.setPixmap(rounded)

    def _load_history(self):
        rows = list(reversed(self.profile.read_journal()))
        self.lbl_count.setText(f"{len(rows)} ÉVÉNEMENT{'S' if len(rows) > 1 else ''}")
        self.table.setRowCount(len(rows))
        for i, (dt, event) in enumerate(rows):
            d = QTableWidgetItem(dt.strftime("%d/%m/%Y"))
            t = QTableWidgetItem(dt.strftime("%H:%M:%S"))
            for it in (d, t):
                it.setFont(QFont(FONT_NUM, 9))
                it.setForeground(QColor(T["muted"]))
            e = QTableWidgetItem(QIcon(dot_pixmap(self._event_color(event), 10)), "  " + event)
            if event.startswith("Regard indiscret"):
                e.setForeground(QColor(T["danger"]))
            self.table.setItem(i, 0, d)
            self.table.setItem(i, 1, t)
            self.table.setItem(i, 2, e)

    def _export_history(self):
        path, _ = QFileDialog.getSaveFileName(self, "Exporter l'historique",
                                              f"zr_shield_{self.profile.name}.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", encoding="utf-8-sig") as f:
            f.write("Date;Heure;Événement\n")
            for dt, event in self.profile.read_journal():
                f.write(f"{dt:%d/%m/%Y};{dt:%H:%M:%S};{event}\n")
        self.toast.notify("Historique exporté", os.path.basename(path), ACCENT, 2500)

    def _clear_history(self):
        if QMessageBox.question(self, "Effacer l'historique",
                                "Supprimer définitivement tous les événements ?") == QMessageBox.Yes:
            self.profile.clear_journal()
            self._load_history()
            self._refresh_activity()

    def _change_password(self):
        new = self.pw_new.text()
        if len(new) < 6:
            msg, ok = "Le nouveau mot de passe doit contenir au moins 6 caractères.", False
        elif new != self.pw_conf.text():
            msg, ok = "Les mots de passe ne correspondent pas.", False
        elif not self.controller.store.change_password(self.profile.name, self.pw_old.text(), new):
            msg, ok = "Mot de passe actuel incorrect.", False
        else:
            msg, ok = "Mot de passe mis à jour.", True
            for ed in (self.pw_old, self.pw_new, self.pw_conf):
                ed.clear()
        self.pw_msg.setText(msg)
        self.pw_msg.setStyleSheet(f"color:{T['accent'] if ok else T['danger']};")

    def _delete_face(self):
        if self.owner is None:
            return
        if QMessageBox.question(self, "Profil facial",
                                "Supprimer votre profil facial ? La protection sera désactivée.") != QMessageBox.Yes:
            return
        self._stop()
        self.owner = None
        self.profile.clear_face()
        self._log("Profil facial supprimé")
        self._refresh_profile()
        self._status_key = None
        self._refresh_status()

    # ---------- fenêtre ----------
    def _show_window(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event):
        if self._quitting or self.state == "idle":
            self.shutdown()
            event.accept()
            QApplication.quit()
            return
        event.ignore()
        self.hide()
        if not self._tray_hint_shown:
            self._tray_hint_shown = True
            self.toast.notify("ZR-SHIELD reste actif", "La protection continue en arrière-plan.", ACCENT, 3000)

    def shutdown(self):
        self.testing = False
        self.calibrating = False
        self._clock.stop()
        if self.state != "idle":
            self._reset_detection()
            self.profile.log("Protection désactivée")
        self.state = "idle"
        self._release_thread()
        self.shields.shutdown()
        self.toast.hide()
        if self._hotkey:
            _user32.UnregisterHotKey(int(self.winId()), HOTKEY_ID)
            self._hotkey = False
        if self._hotkey_pause:
            _user32.UnregisterHotKey(int(self.winId()), HOTKEY_PAUSE_ID)
            self._hotkey_pause = False
        self._save_timer.stop()
        self.profile.save_config()
        self.tray.hide()

    def _quit(self):
        self._quitting = True
        self.close()

    def _logout(self):
        self.controller.logout()


class AppController:
    def __init__(self, app):
        self.app = app
        self.store = UserStore()
        self.window = None

    def login(self):
        # L'écran de connexion reprend le thème du dernier utilisateur.
        theme = "sombre"
        if self.store.last_user:
            theme = UserProfile(self.store.last_user).config.get("theme", "sombre")
        apply_theme(self.app, theme)
        dlg = LoginDialog(self.store)
        if dlg.exec_() != QDialog.Accepted:
            return False
        profile = UserProfile(dlg.username)
        apply_theme(self.app, profile.config.get("theme", "sombre"))
        self.window = MainWindow(profile, self)
        self.window.show()
        return True

    def logout(self):
        if self.window:
            self.window.shutdown()
            self.window.hide()
            self.window.deleteLater()
            self.window = None
        if not self.login():
            self.app.quit()


def main():
    if IS_WINDOWS:
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("BMMSystems.ZRShield")
        except Exception:
            pass
    # Coordonnées en pixels physiques : la zone nette tombe exactement sous le curseur.
    QApplication.setAttribute(Qt.AA_DisableHighDpiScaling)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    resolve_fonts()
    apply_theme(app, "sombre")
    app.setWindowIcon(QIcon(shield_pixmap(64, ACCENT)))
    app.setQuitOnLastWindowClosed(False)

    missing = [os.path.basename(m) for m in (YUNET_MODEL, SFACE_MODEL) if not os.path.exists(m)]
    if missing:
        QMessageBox.critical(None, APP_NAME, "Fichiers manquants dans le dossier models :\n" + "\n".join(missing))
        return 1

    controller = AppController(app)
    if not controller.login():
        return 0
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
