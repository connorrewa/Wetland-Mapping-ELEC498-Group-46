"""
config.py — Configuration for the Wetland Mapping backend.

Update GEOTIFF_PATH to point at your pre-computed predictions GeoTIFF.
Nothing else should need to change.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))

# ── Pre-computed GeoTIFF Directory ────────────────────────────────────────────
# Point this at the directory containing your GeoTIFF files.
GEOTIFF_DIR = r"C:\Users\gavin\.gemini\antigravity\scratch\Wetland-Mapping-ELEC498-Group-46\gui\frontend\tif_files"

# ── Wetland class definitions ─────────────────────────────────────────────────
# Must match CONFIG.WETLAND_CLASSES in frontend/app.js
WETLAND_CLASSES = {
    0: 'Background',
    1: 'Fen (Graminoid)',
    2: 'Fen (Woody)',
    3: 'Marsh',
    4: 'Shallow Open Water',
    5: 'Swamp',
}

VALID_CLASS_MIN = 0
VALID_CLASS_MAX = 5
NODATA_VALUE = 255
