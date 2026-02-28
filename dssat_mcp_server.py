"""
DSSAT-MCP Server (proof-of-concept)

Wraps DSSAT CSM v4.8 as an MCP-compliant service with three tools:
  - list_models()
  - run_simulation(payload: JSON)
  - get_result(job_id: str)

Design goals
- Contract-first schemas (JSON Schema-like dicts embedded in tool metadata)
- Reproducibility (content-addressed job folders + SHA-256 manifest)
- Portability (Windows-native; Linux/macOS via Wine)
- Minimal deps: fastmcp (https://pypi.org/project/fastmcp/), pydantic, python-dotenv (optional)

Quickstart (PowerShell)
1) python -m venv .venv; .venv\\Scripts\\Activate.ps1
2) pip install fastmcp pydantic python-dotenv
3) Set environment variables (see ENV section below) or create .env next to this file.
4) python dssat_mcp_server.py  (starts an MCP server over stdio)
5) Connect from an MCP-capable client (Claude Desktop, etc.) by registering this server.

NOTE: This is a PoC. Adapt file templates (FILEX_TPL, CTRNO handling), station codes, and crop codes
      to your local DSSAT installation. Validate outputs empirically.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from pydantic import BaseModel, Field, ValidationError
except Exception as e:  # pragma: no cover
    print("Pydantic is required: pip install pydantic", file=sys.stderr)
    raise

try:
    from fastmcp import FastMCP  # type: ignore
except Exception as e:  # pragma: no cover
    print("fastmcp is required: pip install fastmcp", file=sys.stderr)
    raise

try:
    from weather_utils import create_weather_station as _kma_download
    _WEATHER_UTILS_OK = True
except ImportError:
    _WEATHER_UTILS_OK = False

# -----------------------------
# ENV & PATHS
# -----------------------------
# Required env vars (create .env file or set in shell):
#   DSSAT_HOME   : Root of DSSAT installation (e.g., C:\\DSSAT48)
#   DSSAT_BIN    : Path to DSCSM048.EXE (e.g., C:\\DSSAT48\\DSCSM048.EXE)
#   DSSAT_WORK   : Writable workspace for jobs (e.g., D:\\dssat_jobs)
# Optional:
#   DSSAT_WEATHER: Directory with .WTH files (e.g., C:\\DSSAT48\\Weather)
#   DSSAT_SOIL   : Directory with .SOL files (e.g., C:\\DSSAT48\\Soil)
#   USE_WINE     : If set to '1', uses wine (Linux/macOS) to run DSSAT

from dotenv import load_dotenv  # type: ignore
load_dotenv()

DSSAT_HOME = Path(os.getenv("DSSAT_HOME", "")).resolve()
DSSAT_BIN = Path(os.getenv("DSSAT_BIN", "")).resolve()
DSSAT_WORK = Path(os.getenv("DSSAT_WORK", str(Path.cwd() / "_jobs"))).resolve()
DSSAT_WEATHER = Path(os.getenv("DSSAT_WEATHER", str(DSSAT_HOME / "Weather"))).resolve()
DSSAT_SOIL = Path(os.getenv("DSSAT_SOIL", str(DSSAT_HOME / "Soil"))).resolve()
USE_WINE = os.getenv("USE_WINE", "0") == "1"

DSSAT_WORK.mkdir(parents=True, exist_ok=True)

# -----------------------------
# SUPPORTED MODELS / CODES (edit to your install)
# -----------------------------
SUPPORTED_CROPS = {
    "maize": {
        "code": "MZ", "cr": "MZ", "ext": "MZX", "model": "MZCER048",
        "ingeno": "KR0003", "cname": "Dacheongok",
        "ppop": 7.5, "plrs": 75.0, "plme": "S",
        "symbi": "N", "cul_file": "MZCER048.CUL",
        "hlast_days": 210,  # max days from sowing to forced harvest
    },
    "wheat": {
        "code": "WH", "cr": "WH", "ext": "WHX", "model": "WHCER048",
        "ingeno": "IB0488", "cname": "NEWTON",
        "ppop": 162.0, "plrs": 16.0, "plme": "S",
        "symbi": "N", "cul_file": "WHCER048.CUL",
        "hlast_days": 260,  # winter wheat ~200-220d; 260 to give buffer
    },
    "rice": {
        "code": "RI", "cr": "RI", "ext": "RIX", "model": "RICER048",
        "ingeno": "IB0012", "cname": "IR 58",
        "ppop": 25.0, "plrs": 20.0, "plme": "T",
        "symbi": "N", "cul_file": "RICER048.CUL",
        "hlast_days": 190,
    },
    "soybean": {
        "code": "SB", "cr": "SB", "ext": "SBX", "model": "CRGRO048",
        "ingeno": "KR2828", "cname": "KRUG2828",
        "ppop": 30.0, "plrs": 60.0, "plme": "S",
        "symbi": "Y", "cul_file": "SBGRO048.CUL",
        "hlast_days": 160,
    },
    "barley": {
        "code": "BA", "cr": "BA", "ext": "BAX", "model": "CSCER048",
        "ingeno": "KR0001", "cname": "Tapgol",
        "ppop": 736.0, "plrs": 20.0, "plme": "S",
        "symbi": "N", "cul_file": "BACER048.CUL",
        "hlast_days": 230,  # fall barley can be 200-220d; 230 buffer
    },
    "potato": {
        "code": "PT", "cr": "PT", "ext": "PTX", "model": "PTSUB048",
        "ingeno": "IB0001", "cname": "MAJESTIC",
        "ppop": 5.5, "plrs": 75.0, "plme": "S",
        "symbi": "N", "cul_file": "PTSUB048.CUL",
        "pldp": 8, "plwt": 1500, "sprl": 2,  # depth (cm), seed tuber weight (g/m2), sprout length (cm)
        "hlast_days": 150,
    },
    "sorghum": {
        "code": "SG", "cr": "SG", "ext": "SGX", "model": "SGCER048",
        "ingeno": "IB0001", "cname": "RIO",
        "ppop": 20.0, "plrs": 75.0, "plme": "S",
        "symbi": "N", "cul_file": "SGCER048.CUL",
        "hlast_days": 170,
    },
}

SUPPORTED_STATIONS = {
    "SUWO": "SUWO2501.WTH",  # Suwon, S. Korea (KMA)
}

SUPPORTED_SOILS = {
    "KR_JD_MAI1": "KR.SOL",  # Suwon Jungdong, SiC 120cm (Maize/Soybean)
    "KR_JD_MAI2": "KR.SOL",  # Suwon Jungdong, SiL 120cm (Maize/Soybean/Cabbage)
}

# -----------------------------
# JSON PAYLOAD SCHEMA
# -----------------------------
class ManagementSpec(BaseModel):
    nitrogen_fertilizer_kg_ha: float = Field(ge=0, description="Total N as basal for PoC")
    irrigation_mm: float = Field(ge=0, description="Total irrigation (mm) applied once for PoC")

class ClimateScenario(BaseModel):
    weather: str = Field(description="Station code key, e.g., ICN01")
    delta_temp: float = Field(default=0.0)
    rainfall_factor: float = Field(default=1.0)

class ControlsSpec(BaseModel):
    ctrno: int = Field(default=1, ge=1, le=99, description="CTRNO to select output verbosity")

class BatchItem(BaseModel):
    # Extend as needed; for PoC we mirror the root fields that matter
    sowing_date: Optional[str] = None
    management: Optional[ManagementSpec] = None
    climate_scenario: Optional[ClimateScenario] = None

class RunPayload(BaseModel):
    crop: str = Field(description="wheat | maize | rice")
    sowing_date: str = Field(description="YYYY-MM-DD")
    soil_profile: str = Field(description="e.g., Suwon_Loam")
    management: ManagementSpec
    climate_scenario: ClimateScenario
    controls: ControlsSpec = Field(default_factory=ControlsSpec)
    batch: List[BatchItem] = Field(default_factory=list, description="Optional multi-scenario list")
    job_id: Optional[str] = Field(default=None, description="If provided, reuse/overwrite job folder")

# (FileX content is built programmatically in write_filex below)

# -----------------------------
# UTILITIES
# -----------------------------

def ymd_to_yyddd(date_str: str) -> Tuple[int, int]:
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    yydoy = int(f"{d.year % 100:02d}{d.timetuple().tm_yday:03d}")
    return d.year, yydoy


def sha256_of_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(job_dir: Path, extra: Dict[str, Any]) -> Dict[str, Any]:
    manifest = {"files": {}, **extra}
    for path in job_dir.rglob("*"):
        if path.is_file():
            rel = str(path.relative_to(job_dir))
            try:
                manifest["files"][rel] = sha256_of_path(path)
            except Exception:
                pass
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def prepare_weather(
    weather_key: str, climate: "ClimateScenario", job_dir: Path, trno: int
) -> Tuple[Path, str, float, float, float]:
    """Locate or create the scenario weather file.

    Accepts any WTH file stem present in DSSAT_WEATHER (e.g., "SUWO2501",
    "AMES9601").  No whitelist required.

    - Unperturbed: returns the original file and its stem as WSTA code.
    - Perturbed (+T / *rain): creates a uniquely-named copy in DSSAT_WEATHER,
      applies the perturbation, and returns that file.
      Caller is responsible for deleting the perturbed file after the run.

    Returns: (wth_path, wsta_code, lat, lon, elev)
    """
    # Dynamic lookup: search DSSAT_WEATHER for "{weather_key}.WTH"
    src: Optional[Path] = None
    candidate = DSSAT_WEATHER / f"{weather_key}.WTH"
    if candidate.exists():
        src = candidate
    else:
        # Case-insensitive fallback (useful on non-Windows)
        key_upper = weather_key.upper()
        for f in sorted(DSSAT_WEATHER.glob("*.WTH")):
            if f.stem.upper() == key_upper:
                src = f
                break

    if src is None:
        raise FileNotFoundError(
            f"Weather file '{weather_key}.WTH' not found in {DSSAT_WEATHER}"
        )

    # Extract lat/lon/elev from WTH header
    hdr = parse_wth_header(src)
    lat, lon, elev = hdr["lat"], hdr["lon"], hdr["elev"]

    is_perturbed = abs(climate.delta_temp) > 1e-9 or abs(climate.rainfall_factor - 1.0) > 1e-9
    if not is_perturbed:
        return src, src.stem, lat, lon, elev  # original file

    # Build a unique 8-char WSTA code from the job directory name and scenario index
    job4 = job_dir.name[:4].upper()          # e.g. "4B1C"
    wsta_code = f"PB{job4}{trno:02d}"        # e.g. "PB4B1C01"  (8 chars)
    dst = DSSAT_WEATHER / f"{wsta_code}.WTH"
    shutil.copy2(src, dst)
    perturb_weather(dst, climate.delta_temp, climate.rainfall_factor)
    return dst, wsta_code, lat, lon, elev


def check_wth_range(wth_stem: str, sow: dt.date, hlast_days: int) -> Optional[str]:
    """Check if a WTH file covers the full simulation period.

    Returns a warning string if data is insufficient, or None if OK.
    Simulation period: (sow - 10 days) through (sow + hlast_days).
    """
    try:
        src: Optional[Path] = None
        candidate = DSSAT_WEATHER / f"{wth_stem}.WTH"
        if candidate.exists():
            src = candidate
        else:
            key_upper = wth_stem.upper()
            for f in sorted(DSSAT_WEATHER.glob("*.WTH")):
                if f.stem.upper() == key_upper:
                    src = f
                    break
        if src is None:
            return None  # file not found; prepare_weather will raise later

        hdr = parse_wth_header(src, read_data_range=True)
        data_start_str = hdr.get("data_start", "")
        data_end_str   = hdr.get("data_end",   "")
        if not data_start_str or not data_end_str:
            return None  # can't determine range

        data_start = dt.date.fromisoformat(data_start_str)
        data_end   = dt.date.fromisoformat(data_end_str)
        sim_start  = sow - dt.timedelta(days=10)
        sim_end    = sow + dt.timedelta(days=hlast_days)

        msgs: List[str] = []
        if sim_start < data_start:
            msgs.append(
                f"Simulation start {sim_start} is before WTH data start {data_start_str}. "
                f"Results may be unreliable."
            )
        if sim_end > data_end:
            msgs.append(
                f"Maximum harvest window ends {sim_end} but WTH data ends {data_end_str}. "
                f"If the crop matures before {data_end_str} simulation will still succeed; "
                f"otherwise use a WTH file covering through {sim_end}."
            )
        return " ".join(msgs) if msgs else None
    except Exception:
        return None


def ensure_soil(soil_key: str, job_dir: Path) -> Path:
    src = find_sol_file(soil_key)  # raises FileNotFoundError if not found
    dst = job_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def parse_cul_file(cul_path: Path) -> List[Dict[str, str]]:
    """Parse a DSSAT .CUL file and return list of {var_id, name} dicts.

    Handles both CERES (@VAR# VRNAME.......... ...) and CROPGRO
    (@VAR# VAR-NAME........ ...) header formats by locating the EXPNO/EXP#
    column position to determine where the name field ends.
    """
    cultivars: List[Dict[str, str]] = []
    name_start = 7   # VAR# is always 6 chars; name starts at position 7
    name_end = 25    # default fallback

    for line in cul_path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Header line: locate where the name field ends (before EXPNO/EXP#)
        if stripped.startswith("@VAR"):
            upper = line.upper()
            for marker in ("EXPNO", "EXP#"):
                idx = upper.find(marker)
                if idx != -1:
                    name_end = idx
                    break
            continue
        # Skip comments and section headers
        if stripped.startswith("!") or stripped.startswith("*"):
            continue
        # Data line: first char is alphanumeric
        if line and line[0].isalnum():
            var_id = line[0:6].strip()
            name = line[name_start:name_end].strip() if len(line) > name_start else ""
            # Skip min/max bound entries (999991, 999992)
            if var_id and not var_id.startswith("9999"):
                cultivars.append({"var_id": var_id, "name": name})

    return cultivars


def lookup_cultivar(crop: str, var_id: str) -> Tuple[str, str]:
    """Return (ingeno, cname) for a given cultivar ID, or raise ValueError.

    Searches the crop's .CUL file in DSSAT_HOME/Genotype/.
    """
    cul_path = DSSAT_HOME / "Genotype" / SUPPORTED_CROPS[crop]["cul_file"]
    if not cul_path.exists():
        raise FileNotFoundError(f"CUL file not found: {cul_path}")
    for entry in parse_cul_file(cul_path):
        if entry["var_id"].upper() == var_id.upper():
            return entry["var_id"], entry["name"]
    raise ValueError(f"Cultivar '{var_id}' not found in {cul_path.name}")


def parse_sol_profile(soil_key: str) -> List[Dict[str, str]]:
    """Parse layer data for a single soil profile from any DSSAT .SOL file.

    Searches all .SOL files in DSSAT_SOIL (SUPPORTED_SOILS fast-path first).
    Returns a list of dicts, one per soil layer, keyed by the column names
    from the @SLB header line (SLB, SLMH, SLLL, SDUL, SSAT, SRGF, …).
    """
    sol_path = find_sol_file(soil_key)  # raises FileNotFoundError if not found

    layers: List[Dict[str, str]] = []
    in_profile = False
    layer_col_names: List[str] = []

    for raw_line in sol_path.read_text(errors="ignore").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        # Start of the target profile: "*KR_JD_MAI1  ..." (exact ID match)
        if stripped.startswith(f"*{soil_key}") and (
            len(stripped) == len(soil_key) + 1
            or stripped[len(soil_key) + 1] in (" ", "\t")
        ):
            in_profile = True
            layer_col_names = []
            layers = []
            continue

        # Start of any other profile section → stop
        if in_profile and stripped.startswith("*") and not stripped.startswith("*RUN"):
            break

        if not in_profile:
            continue

        # Layer column header: starts with "@" and contains "SLB"
        if stripped.startswith("@") and "SLB" in stripped.upper():
            layer_col_names = stripped.lstrip("@ ").split()
            continue

        # Layer data row: first token is a number (SLB depth)
        if layer_col_names and stripped and stripped[0].isdigit():
            parts = stripped.split()
            row = {col: (parts[i] if i < len(parts) else "-99")
                   for i, col in enumerate(layer_col_names)}
            layers.append(row)

    return layers


def perturb_weather(wth_path: Path, delta_temp: float, rainfall_factor: float) -> None:
    """Perturb a DSSAT .WTH file: add delta_temp to TMAX/TMIN, scale RAIN.

    Standard DSSAT weather column layout (6-char fields after the 5-char DATE):
      Positions   Column
      [0:5]       DATE  (YYDDD)
      [5:11]      SRAD  — left unchanged
      [11:17]     TMAX
      [17:23]     TMIN
      [23:29]     RAIN
      [29:]       remaining columns (DEWP, WIND, PAR, …) — left unchanged

    Previous code used wrong offsets (6:12, 13:19, 20:26) which read
    SRAD as TMAX, TMAX as TMIN, and part of TMIN (dropping the minus sign)
    as RAIN.  This version uses the correct offsets validated against
    SUWO2501.WTH.
    """
    lines = wth_path.read_text(errors="replace").splitlines()
    out = []
    in_data = False
    for ln in lines:
        if not in_data and ln.strip().startswith("@DATE"):
            in_data = True
            out.append(ln)
            continue
        if in_data and ln.strip() and ln[0].isdigit():
            try:
                date = ln[0:5]
                srad = ln[5:11]                          # SRAD — unchanged
                tmax = float(ln[11:17]) + delta_temp
                tmin = float(ln[17:23]) + delta_temp
                rain = max(0.0, float(ln[23:29]) * rainfall_factor)
                rest = ln[29:]                           # DEWP, WIND, etc.
                out.append(f"{date}{srad}{tmax:6.1f}{tmin:6.1f}{rain:6.1f}{rest}")
            except Exception:
                out.append(ln)
        else:
            out.append(ln)
    wth_path.write_text("\n".join(out))


def parse_wth_header(wth_path: Path, read_data_range: bool = False) -> Dict[str, Any]:
    """Parse the header block of a DSSAT .WTH file.

    Returns dict: {insi, lat, lon, elev, tav, amp}
    If read_data_range=True, also returns {data_start, data_end} as YYYY-MM-DD strings.
    Falls back to 0.0 / "" for fields that are missing or malformed.
    """
    result: Dict[str, Any] = {
        "insi": wth_path.stem[:4].upper(),
        "lat": 0.0, "lon": 0.0, "elev": 0.0, "tav": 0.0, "amp": 0.0,
    }
    try:
        lines = wth_path.read_text(errors="ignore").splitlines()
        in_data = False
        first_yyddd: Optional[str] = None
        last_yyddd: Optional[str] = None

        for i, ln in enumerate(lines):
            stripped = ln.strip()
            # Header descriptor line: "@ INSI    LAT    LONG   ELEV ..."
            if stripped.startswith("@ INSI") or stripped.upper().startswith("@INSI"):
                for j in range(i + 1, min(i + 6, len(lines))):
                    val = lines[j].strip()
                    if val and not val.startswith("!") and not val.startswith("@"):
                        parts = val.split()
                        if len(parts) >= 4:
                            result["insi"] = parts[0]
                            try:
                                result["lat"]  = float(parts[1])
                                result["lon"]  = float(parts[2])
                                result["elev"] = float(parts[3])
                                if len(parts) >= 5: result["tav"] = float(parts[4])
                                if len(parts) >= 6: result["amp"] = float(parts[5])
                            except ValueError:
                                pass
                        break
                if not read_data_range:
                    break  # header-only mode: stop here
                continue

            if not read_data_range:
                continue

            # Data section starts after @DATE header
            if stripped.startswith("@DATE"):
                in_data = True
                continue
            if in_data and stripped and stripped[0].isdigit() and len(stripped) >= 5:
                yyddd = stripped[:5]
                if first_yyddd is None:
                    first_yyddd = yyddd
                last_yyddd = yyddd

        if read_data_range:
            result["data_start"] = _yyddd_to_iso(first_yyddd) if first_yyddd else ""
            result["data_end"]   = _yyddd_to_iso(last_yyddd)  if last_yyddd  else ""
    except Exception:
        pass
    return result


def _yyddd_to_iso(yyddd: str) -> str:
    """Convert DSSAT YYDDD string to ISO date. YY 00-30 → 2000+, 31-99 → 1900+."""
    try:
        yy = int(yyddd[:2])
        doy = int(yyddd[2:5])
        year = (2000 + yy) if yy <= 30 else (1900 + yy)
        return (dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)).isoformat()
    except Exception:
        return ""


def _dssatdate_to_iso(s: str) -> str:
    """Convert DSSAT summary date string (YYYYDDD 7-char or YYDDD 5-char) to ISO YYYY-MM-DD."""
    try:
        s = str(s).strip()
        if len(s) == 7:          # YYYYDDD (Summary.OUT format)
            year = int(s[:4]); doy = int(s[4:])
        elif len(s) == 5:        # YYDDD (WTH file format)
            yy = int(s[:2]); doy = int(s[2:])
            year = (2000 + yy) if yy <= 30 else (1900 + yy)
        else:
            return s
        if doy <= 0:
            return s
        return (dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)).isoformat()
    except Exception:
        return s


def _make_human_summary(run: Dict[str, Any], crop_name: str) -> Dict[str, Any]:
    """Build a human-readable interpretation of one DSSAT simulation run row."""
    def _date(k: str) -> str:
        v = run.get(k, "")
        return _dssatdate_to_iso(v) if v and v not in ("-99", "") else None

    def _num(k: str) -> Any:
        v = str(run.get(k, "-99")).strip()
        if v in ("-99", "", "—", "None"):
            return None
        try:
            f = float(v)
            return int(f) if f == int(f) else round(f, 2)
        except Exception:
            return v

    pdat = _date("PDAT"); mdat = _date("MDAT")
    try:
        growing_days = (dt.date.fromisoformat(mdat) - dt.date.fromisoformat(pdat)).days
    except Exception:
        growing_days = None

    hwam = _num("HWAM")
    cwam = _num("CWAM")
    etcm = _num("ETCM")
    nicm = _num("NICM")
    ircm = _num("IRCM")

    # Derived agronomic indicators
    try:
        hi = round(float(hwam) / float(cwam), 3) if hwam and cwam and float(cwam) > 0 else None
    except Exception:
        hi = None
    try:
        wue = round(float(hwam) / float(etcm), 2) if hwam and etcm and float(etcm) > 0 else None
    except Exception:
        wue = None
    try:
        nue = round(float(hwam) / float(nicm), 1) if hwam and nicm and float(nicm) > 0 else None
    except Exception:
        nue = None

    return {
        "crop": crop_name,
        "sowing_date":                    pdat,
        "emergence_date":                 _date("EDAT"),
        "anthesis_date":                  _date("ADAT"),
        "maturity_date":                  mdat,
        "growing_days":                   growing_days,
        "grain_yield_kg_ha":              hwam,
        "total_biomass_kg_ha":            cwam,
        "harvest_index":                  hi,
        "seasonal_rainfall_mm":           _num("PRCM"),
        "evapotranspiration_mm":          etcm,
        "irrigation_applied_mm":          ircm,
        "water_use_efficiency_kg_mm":     wue,
        "n_uptake_kg_ha":                 nicm,
        "n_use_efficiency_kg_kg":         nue,
    }


def find_sol_file(soil_key: str) -> Path:
    """Search all .SOL files in DSSAT_SOIL for a profile with the given ID.

    Checks SUPPORTED_SOILS first for fast lookup, then falls back to a full
    scan of every .SOL file in the directory.

    Returns the Path to the matching .SOL file, or raises FileNotFoundError.
    """
    # Fast path: legacy SUPPORTED_SOILS mapping
    if soil_key in SUPPORTED_SOILS:
        candidate = DSSAT_SOIL / SUPPORTED_SOILS[soil_key]
        if candidate.exists():
            return candidate

    # Full scan of all SOL files
    for sol_path in sorted(DSSAT_SOIL.glob("*.SOL")):
        try:
            for line in sol_path.read_text(errors="ignore").splitlines():
                s = line.strip()
                if s.startswith(f"*{soil_key}") and (
                    len(s) == len(soil_key) + 1
                    or s[len(soil_key) + 1] in (" ", "\t")
                ):
                    return sol_path
        except Exception:
            continue

    raise FileNotFoundError(
        f"Soil profile '{soil_key}' not found in any .SOL file under {DSSAT_SOIL}"
    )


def write_filex(
    payload: RunPayload,
    job_dir: Path,
    trno: int = 1,
    wsta: str = "",
    cultivar_id: Optional[str] = None,
    cname: Optional[str] = None,
    ppop_override: Optional[float] = None,
    plrs_override: Optional[float] = None,
    fert_dap: int = 14,
    irr_dap: int = 14,
    lat: float = 0.0,
    lon: float = 0.0,
    elev: float = 0.0,
    site_name: str = "",
    co2_ppm: Optional[float] = None,
    fertilizer_events: Optional[List[Dict[str, Any]]] = None,
    ic_wr: Optional[float] = None,
    irrigation_events: Optional[List[Dict[str, Any]]] = None,
    auto_irrigate: bool = False,
    irr_threshold: float = 50.0,
    irr_target: float = 100.0,
    irr_depth_cm: int = 30,
    irr_amount_mm: float = 30.0,
    irr_method: str = "IR001",
    tillage_events: Optional[List[Dict[str, Any]]] = None,
    residue_events: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Write a DSSAT FileX experiment file.

    Args:
        cultivar_id: Override the default cultivar VAR# (e.g. "IB0001").
        cname:        Override the cultivar name (looked up automatically if omitted).
        ppop_override: Plant population (plants/m²); uses crop default if None.
        plrs_override: Row spacing (cm); uses crop default if None.
        ic_wr: Initial soil water fraction (0=wilting point, 1=field capacity). Default 0.7.
        irrigation_events: List of irrigation events for scheduled irrigation.
               Each dict must contain: dap (days after planting, int), amount_mm (float).
               Optional key: irop (irrigation method code, default "IR001").
               Overrides irrigation_mm + irr_dap when provided.
        auto_irrigate: If True, DSSAT applies irrigation automatically when soil water
               drops below irr_threshold. Uses IRRIG=A in simulation controls.
               Takes priority over irrigation_events and irrigation_mm.
        irr_threshold: Soil water threshold (% AWC) at which auto-irrigation triggers. Default 50.
        irr_target: Soil water target (% AWC) to refill to after auto-irrigation. Default 100.
        irr_depth_cm: Soil depth (cm) monitored for auto-irrigation trigger. Default 30.
        irr_amount_mm: Amount applied per auto-irrigation event (mm). Default 30.
        irr_method: DSSAT irrigation method code (IR001=sprinkler, IR004=drip). Default IR001.
        fert_dap:     Days after planting for basal fertiliser application. Default 14.
        irr_dap:      Days after planting for supplemental irrigation. Default 14.
        co2_ppm:      Atmospheric CO2 concentration (ppm) override. Default None = 425 ppm.
                      E.g. 550 for RCP4.5 mid-century, 700 for RCP8.5 end-century.
        fertilizer_events: List of fertilizer events, each a dict with keys:
                      dap (days after planting, required), n_kg_ha (N amount, default 0),
                      p_kg_ha (P amount, optional), k_kg_ha (K amount, optional).
                      If None, falls back to payload.management.nitrogen_fertilizer_kg_ha + fert_dap.
        tillage_events: List of tillage operations. Each dict must contain:
                      dap (int, days after planting; negative = before planting),
                      implement (str, DSSAT code e.g. "TI007" = disk plow; default "TI009" tandem disk).
                      Optional keys: depth_cm (float, tillage depth in cm; default -99 = implement default),
                      name (str, label; default "-99").
                      Common codes: TI001=V-Ripper, TI003=Moldboard 20cm, TI004=Chisel sweeps,
                      TI007=Disk plow, TI009=Tandem disk, TI014=Spike harrow.
                      If None, no tillage section is written (TILL=N).
        residue_events: List of crop residue / organic fertilizer applications.
                      Each dict must contain: dap (int, days after planting; negative = before),
                      amount_kg_ha (float, dry matter amount in kg/ha).
                      Optional keys:
                        rcod (str): residue/manure code. Default "RE001" (maize residue).
                          Residue: RE001=maize, RE002=wheat, RE003=rice, RE004=soybean.
                          Manure: RF001=farmyard manure, RF002=poultry manure.
                        resn_pct (float): N concentration %. Default 1.5.
                        resp_pct (float): P concentration %. Default 0.2.
                        resk_pct (float): K concentration %. Default 0.0.
                        incorporation_pct (float): % incorporated into soil (0=surface mulch, 100=fully mixed). Default 0.
                        depth_cm (float): incorporation depth (cm). Default 20.
                      Example (surface mulch): [{"dap": -7, "rcod": "RE002", "amount_kg_ha": 3000, "incorporation_pct": 0}]
                      Example (manure): [{"dap": -14, "rcod": "RF001", "amount_kg_ha": 5000, "resn_pct": 0.5, "incorporation_pct": 100}]
    """
    crop = SUPPORTED_CROPS[payload.crop]
    year, pdate = ymd_to_yyddd(payload.sowing_date)
    sow = dt.datetime.strptime(payload.sowing_date, "%Y-%m-%d").date()

    def _doy(d: dt.date) -> int:
        _, v = ymd_to_yyddd(d.strftime("%Y-%m-%d"))
        return v

    # Resolve cultivar — override > SUPPORTED_CROPS default
    ingeno = cultivar_id or crop["ingeno"]
    if cname is None and cultivar_id is not None and cultivar_id != crop["ingeno"]:
        # Look up name from CUL file when a non-default cultivar is specified
        try:
            _, cname = lookup_cultivar(payload.crop, cultivar_id)
        except Exception:
            cname = cultivar_id  # fallback: use the ID as name
    cname = cname or crop["cname"]

    # Resolve agronomic defaults
    ppop = ppop_override if ppop_override is not None else crop["ppop"]
    plrs = plrs_override if plrs_override is not None else crop["plrs"]

    # Key dates
    sdate = _doy(sow - dt.timedelta(days=10))             # simulation start (10 d before sowing)
    icdat = sdate                                          # initial conditions same as sim start
    fdate = _doy(sow + dt.timedelta(days=fert_dap))       # fertiliser / irrigation application
    irr_date = _doy(sow + dt.timedelta(days=irr_dap))     # irrigation (may differ from fert)
    pfrst = _doy(sow - dt.timedelta(days=7))              # auto-plant window open
    plast = _doy(sow + dt.timedelta(days=7))              # auto-plant window close
    hlast = _doy(sow + dt.timedelta(days=crop.get("hlast_days", 200)))  # crop-specific harvest window

    # Build fertilizer events list (multi-event override or legacy single-event fallback)
    if fertilizer_events:
        fert_events_list = fertilizer_events
    elif payload.management.nitrogen_fertilizer_kg_ha > 0:
        fert_events_list = [{"dap": fert_dap, "n_kg_ha": payload.management.nitrogen_fertilizer_kg_ha}]
    else:
        fert_events_list = []

    # Determine irrigation mode and build event list
    if auto_irrigate:
        irr_mode       = "A"           # automatic: threshold-based
        irr_events_list: List[Dict] = []
    elif irrigation_events:
        irr_mode       = "R"           # reported: explicit schedule
        irr_events_list = irrigation_events
    elif payload.management.irrigation_mm > 0:
        irr_mode       = "R"           # legacy single-event
        irr_events_list = [{"dap": irr_dap, "amount_mm": payload.management.irrigation_mm}]
    else:
        irr_mode       = "N"           # no irrigation
        irr_events_list = []

    has_irr  = irr_mode in ("A", "R")
    has_fert = bool(fert_events_list)
    mi = 1 if has_irr  else 0
    mf = 1 if has_fert else 0
    me = 1 if co2_ppm is not None else 0       # environment factor level (CO2 override)
    mt        = 1 if tillage_events else 0       # tillage factor level
    till_flag = "Y" if tillage_events else "N"   # TILL option flag
    mr        = 1 if residue_events else 0       # residue/organic fert factor level

    exp_name = f"{crop['code']}{year % 100:02d}{trno:04d}"
    tname    = f"{payload.crop.upper()} T{trno:02d}"
    sname    = f"{payload.crop.upper()}_{year}_{trno:02d}"

    L: List[str] = []

    # --- *GENERAL ---
    L += [
        f"*EXP.DETAILS: {exp_name}",
        "",
        "*GENERAL",
        "@PEOPLE",
        "-99",
        "@ADDRESS",
        "-99",
        "@SITE",
        f" {site_name or wsta[:4]}",
        "@ PAREA  PRNO  PLEN  PLDR  PLSP  PLAY HAREA  HRNO  HLEN  HARM.........",
        "    -99   -99   -99   -99   -99   -99   -99   -99   -99   -99",
        "",
    ]

    # --- *TREATMENTS ---
    L += [
        "*TREATMENTS                        -------------FACTOR LEVELS------------",
        "@N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM",
        f" 1 1 0 0 {tname:<25}  1  1  0  1  1  {mi}  {mf}  {mr}  0  {mt}  {me}  0  1",
        "",
    ]

    # --- *CULTIVARS ---
    L += [
        "*CULTIVARS",
        "@C CR INGENO CNAME",
        f" 1 {crop['cr']} {ingeno} {cname}",
        "",
    ]

    # --- *FIELDS ---
    field_id = f"{wsta[:4]}0001"  # e.g. "SUWO0001" derived from station code
    L += [
        "*FIELDS",
        "@L ID_FIELD WSTA....  FLSA  FLOB  FLDT  FLDD  FLDS  FLST SLTX  SLDP  ID_SOIL    FLNAME",
        f" 1 {field_id:<8} {wsta[:8]:<8}   -99     0 DR000     0     0 00000 -99    120  {payload.soil_profile:<10} -99",
        "@L ...........XCRD ...........YCRD .....ELEV .............AREA .SLEN .FLWR .SLAS FLHST FHDUR",
        f" 1         {lon:16.3f}        {lat:16.3f} {elev:9.0f}                 0     0     0     0   -99   -99",
        "",
    ]

    # --- *INITIAL CONDITIONS (derived from soil profile via parse_sol_profile) ---
    sol_layers = parse_sol_profile(payload.soil_profile)
    ic_layer_lines: List[str] = []
    for layer in sol_layers:
        slb_val = int(float(layer.get("SLB", "20")))
        sdul = float(layer.get("SDUL", "0.25"))
        slll = float(layer.get("SLLL", "0.15"))
        ic_frac = max(0.0, min(1.0, ic_wr)) if ic_wr is not None else 0.7
        sh2o = slll + ic_frac * (sdul - slll)       # initial SH2O: fraction of AWC (0=WP, 1=FC)
        snh4 = 0.5                                  # ammonium-N (kg/ha), uniform default
        sno3 = round(max(1.0, 5.0 * (1.0 - slb_val / 200.0)), 1)  # nitrate-N decreasing with depth
        ic_layer_lines.append(f" 1  {slb_val:4d}  {sh2o:.3f}   {snh4:.1f}   {sno3:.1f}")
    L += [
        "*INITIAL CONDITIONS",
        "@C   PCR ICDAT  ICRT  ICND  ICRN  ICRE  ICWD ICRES ICREN ICREP ICRIP ICRID ICNAME",
        f" 1    {crop['cr']} {icdat:5d}   600   -99     1     1   -99     0     0     0   100    15 -99",
        "@C  ICBL  SH2O  SNH4  SNO3",
        *ic_layer_lines,
        "",
    ]

    # --- *PLANTING DETAILS ---
    pldp_val = crop.get("pldp", 5)    # planting depth (cm); default 5, potato uses 8
    plwt_val = crop.get("plwt", -99)  # seed/tuber weight (g/m2); -99 = not specified
    sprl_val = crop.get("sprl", -99)  # sprout length (cm); required >0 for potato
    plwt_str = f"{plwt_val:6.0f}" if plwt_val != -99 else "   -99"
    sprl_str = f"{sprl_val:4.0f}" if sprl_val != -99 else "  -99"
    L += [
        "*PLANTING DETAILS",
        "@P PDATE EDATE  PPOP  PPOE  PLME  PLDS  PLRS  PLRD  PLDP  PLWT  PAGE  PENV  PLPH  SPRL                        PLNAME",
        f" 1 {pdate:5d}   -99  {ppop:4g}   -99     {crop['plme']}     R  {plrs:4.0f}     0  {pldp_val:4d}{plwt_str}   -99   -99   -99  {sprl_str}                        -99",
        "",
    ]

    # --- *IRRIGATION (schedule or auto-irrigate management header) ---
    if irr_mode == "A":
        # Automatic irrigation: DSSAT decides when to irrigate based on @N IRRIGATION thresholds.
        # The *IRRIGATION section here sets management parameters; @I events are empty.
        L += [
            "*IRRIGATION AND WATER MANAGEMENT",
            "@I  EFIR  IDEP  ITHR  IEPT  IOFF  IAME  IAMT IRNAME",
            f" 1   1.0  {irr_depth_cm:4d}  {irr_threshold:4.0f}  {irr_target:4.0f} GS000 {irr_method}  {irr_amount_mm:4.0f} -99",
            "@I IDATE  IROP IRVAL",
            "",
        ]
    elif irr_mode == "R" and irr_events_list:
        # Scheduled irrigation: explicit date/amount events
        irr_lines: List[str] = [
            "*IRRIGATION AND WATER MANAGEMENT",
            "@I  EFIR  IDEP  ITHR  IEPT  IOFF  IAME  IAMT IRNAME",
            " 1   1.0   -99   -99   -99   -99   -99   -99 -99",
            "@I IDATE  IROP IRVAL",
        ]
        for ev in irr_events_list:
            ev_date  = _doy(sow + dt.timedelta(days=int(ev.get("dap", irr_dap))))
            ev_amt   = float(ev.get("amount_mm", payload.management.irrigation_mm))
            ev_irop  = ev.get("irop", "IR001")
            irr_lines.append(f" 1 {ev_date:5d} {ev_irop} {ev_amt:6.0f}")
        irr_lines.append("")
        L += irr_lines

    # --- *FERTILIZERS (only if fertilizer events present) ---
    if has_fert:
        fert_lines: List[str] = [
            "*FERTILIZERS (INORGANIC)",
            "@F FDATE  FMCD  FACD  FDEP  FAMN  FAMP  FAMK  FAMC  FAMO  FOCD FERNAME",
        ]
        for ev in fert_events_list:
            ev_date = _doy(sow + dt.timedelta(days=int(ev.get("dap", fert_dap))))
            n_amt   = float(ev.get("n_kg_ha", 0))
            p_raw   = ev.get("p_kg_ha", None)
            k_raw   = ev.get("k_kg_ha", None)
            p_str   = f"{float(p_raw):5.1f}" if p_raw is not None else "  -99"
            k_str   = f"{float(k_raw):5.1f}" if k_raw is not None else "  -99"
            fert_lines.append(
                f" 1 {ev_date:5d} FE001 AP001    15  {n_amt:5.1f}{p_str}{k_str}   -99   -99   -99 -99"
            )
        fert_lines.append("")
        L += fert_lines

    # --- *TILLAGE AND ROTATIONS (only if tillage events present) ---
    if tillage_events:
        till_lines: List[str] = [
            "*TILLAGE AND ROTATIONS",
            "@T TDATE TIMPL  TDEP TNAME",
        ]
        for ev in tillage_events:
            ev_date   = _doy(sow + dt.timedelta(days=int(ev.get("dap", -7))))
            ev_impl   = ev.get("implement", "TI009")   # default: tandem disk
            ev_depth  = ev.get("depth_cm", None)
            ev_name   = ev.get("name", "-99")
            depth_str = f"{float(ev_depth):5.0f}" if ev_depth is not None else "  -99"
            till_lines.append(f" 1 {ev_date:5d} {ev_impl} {depth_str} {ev_name}")
        till_lines.append("")
        L += till_lines

    # --- *RESIDUES AND ORGANIC FERTILIZER (only if residue events present) ---
    if residue_events:
        res_lines: List[str] = [
            "*RESIDUES AND ORGANIC FERTILIZER",
            "@F RDATE  RCOD  RAMT  RESN  RESP  RESK  RINP  RDEP  RMET RENAME",
        ]
        for ev in residue_events:
            ev_date  = _doy(sow + dt.timedelta(days=int(ev.get("dap", -7))))
            ev_rcod  = ev.get("rcod", "RE001")
            ev_ramt  = float(ev.get("amount_kg_ha", 1000))
            ev_resn  = float(ev.get("resn_pct", 1.5))
            ev_resp  = float(ev.get("resp_pct", 0.2))
            ev_resk  = float(ev.get("resk_pct", 0.0))
            ev_rinp  = float(ev.get("incorporation_pct", 0))
            ev_rdep  = float(ev.get("depth_cm", 20))
            res_lines.append(
                f" 1 {ev_date:5d} {ev_rcod}  {ev_ramt:5.0f}  {ev_resn:4.2f}  {ev_resp:4.2f}  {ev_resk:4.2f}"
                f"  {ev_rinp:4.0f}  {ev_rdep:4.0f} AP001 -99"
            )
        res_lines.append("")
        L += res_lines

    # --- *ENVIRONMENT MODIFICATIONS (only if CO2 override is specified) ---
    if co2_ppm is not None:
        eco2_str = f"R{co2_ppm:4.0f}"   # e.g. "R 550" or "R 700"
        L += [
            "*ENVIRONMENT MODIFICATIONS",
            "@E ODATE EDAY  ERAD  EMAX  EMIN  ERAIN ECO2  EDEW  EWIND ENVNAME  ",
            f" 1 00001 A   0 A   0 A   0 A   0 A   0 {eco2_str} A   0 A   0 {co2_ppm:.0f}ppm_CO2",
            "",
        ]

    # --- *SIMULATION CONTROLS ---
    water_flag = "Y"
    symbi_flag = crop.get("symbi", "N")   # "Y" for soybean (N fixation)
    # SYMBI=Y requires NITRO=Y for CROPGRO to run N-fixation properly
    nitro_flag = "Y" if (has_fert or symbi_flag == "Y") else "N"
    L += [
        "*SIMULATION CONTROLS",
        "@N GENERAL     NYERS NREPS START SDATE RSEED SNAME.................... SMODEL",
        f" 1 GE              1     1     S {sdate:5d}  2150 {sname:<24}",
        "@N OPTIONS     WATER NITRO SYMBI PHOSP POTAS DISES  CHEM  TILL   CO2",
        f" 1 OP              {water_flag}     {nitro_flag}     {symbi_flag}     N     N     N     N     {till_flag}     M",
        "@N METHODS     WTHER INCON LIGHT EVAPO INFIL PHOTO HYDRO NSWIT MESOM MESEV MESOL",
        " 1 ME              M     M     E     R     S     C     R     1     G     R     2",
        "@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS",
        f" 1 MA              R     {irr_mode}     R     N     M",
        "@N OUTPUTS     FNAME OVVEW SUMRY FROPT GROUT CAOUT WAOUT NIOUT MIOUT DIOUT VBOSE CHOUT OPOUT FMOPT",
        " 1 OU              N     Y     Y     1     Y     N     Y     Y     N     N     Y     N     N     A",
        "",
        "@  AUTOMATIC MANAGEMENT",
        "@N PLANTING    PFRST PLAST PH2OL PH2OU PH2OD PSTMX PSTMN",
        f" 1 PL          {pfrst:5d} {plast:5d}    40   100    30    40    10",
        "@N IRRIGATION  IMDEP ITHRL ITHRU IROFF IMETH IRAMT IREFF",
        f" 1 IR            {irr_depth_cm:4d}  {irr_threshold:4.0f}  {irr_target:4.0f} GS000 {irr_method}  {irr_amount_mm:4.0f}     1",
        "@N NITROGEN    NMDEP NMTHR NAMNT NCODE NAOFF",
        " 1 NI             30    50    25 FE001 GS000",
        "@N RESIDUES    RIPCN RTIME RIDEP",
        " 1 RE            100     1    20",
        "@N HARVEST     HFRST HLAST HPCNP HPCNR",
        f" 1 HA              0 {hlast:5d}   100     0",
        "",
    ]

    ext = crop["ext"]
    x_path = job_dir / f"{crop['code']}{year % 100:02d}{trno:04d}.{ext}"
    x_path.write_text("\n".join(L))
    return x_path


def write_dssbatch(filex_paths: List[Path], job_dir: Path, crop_name: str = "") -> Path:
    # Header: $BATCH line + column header matching DSSAT v4.8 fixed-width format
    batch_tag = f"$BATCH({crop_name.upper()})" if crop_name else "$BATCH"
    col_header = f"{'@FILEX':<94}{'TRTNO':>5}{'RP':>7}{'SQ':>7}{'OP':>7}{'CO':>7}"
    lines = [batch_tag, col_header]
    for fx in filex_paths:
        # Full absolute path required; SQ=0 (not a sequence run)
        lines.append(f"{str(fx):<94}{1:5d}{1:7d}{0:7d}{1:7d}{0:7d}")
    path = job_dir / "DSSBatch.v48"
    path.write_text("\n".join(lines))
    return path


def run_dssat(job_dir: Path, model_code: str = "") -> subprocess.CompletedProcess:
    # Command: DSCSM048.EXE [model_code] B DSSBatch.v48
    # model_code is optional (e.g., "MZCER048"); B = batch run mode
    exe = str(DSSAT_BIN)
    base = ["wine", exe] if USE_WINE else [exe]
    args = base + ([model_code] if model_code else []) + ["B", "DSSBatch.v48"]
    return subprocess.run(args, cwd=job_dir, env=os.environ.copy(), capture_output=True, text=True, check=False)


def parse_summary_out(job_dir: Path) -> Dict[str, Any]:
    """Parse Summary.OUT using column positions from the header line.

    Returns a dict with:
      - "runs": list of dicts (compact key columns only)
      - "all_runs": list of dicts (all columns, raw string values)
      - "n_runs": number of run rows found
    """
    # Key agronomic columns to include in the compact view
    KEY_COLS = [
        "RUNNO", "TRNO", "CR", "PDAT", "EDAT", "ADAT", "MDAT", "HDAT",
        "HWAM", "HWAH", "CWAM", "BWAM",
        "IRCM", "PRCM", "ETCM", "EPCE",
        "NICM", "NLCM", "NIAM",
        "DMPPM", "DMPEM", "DMPTM", "DMPIM",
        "CRST",
    ]

    p = job_dir / "Summary.OUT"
    if not p.exists():
        return {"warning": "Summary.OUT missing"}
    text = p.read_text(errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]

    header_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("@") and "RUNNO" in ln.upper():
            header_idx = i
            break
    if header_idx is None:
        return {"warning": "RUN header not found in Summary.OUT"}

    header = lines[header_idx]
    # Parse column boundaries from header.
    #
    # DSSAT Summary.OUT aligns data values so that each value ENDS at the same
    # character position as the last character of its column header token (including
    # any trailing '.' padding in token names like "TNAM.....................").
    # Column i therefore spans data[col_ends[i-1] : col_ends[i]] (exclusive).
    #
    # Using col_starts[i] as the left boundary (the naive approach) fails for
    # columns whose values are wider than their header token (e.g. SDAT is 4-char
    # token but value is 7-char "2025111").
    col_ends: List[int] = []
    col_names: List[str] = []
    i = 0
    while i < len(header):
        if header[i] not in (" ", "\t"):
            start = i
            while i < len(header) and header[i] not in (" ", "\t"):
                i += 1
            name = header[start:i].lstrip("@").rstrip(".")
            col_ends.append(i)   # exclusive end of this token (= right boundary of column)
            col_names.append(name)
        else:
            i += 1

    # Parse all data rows (skip comment/header lines)
    all_runs: List[Dict[str, str]] = []
    for ln in lines[header_idx + 1:]:
        stripped = ln.strip()
        if not stripped or stripped.startswith("!") or stripped.startswith("@"):
            continue
        row: Dict[str, str] = {}
        for j, (name, right) in enumerate(zip(col_names, col_ends)):
            left = col_ends[j - 1] if j > 0 else 0
            row[name] = ln[left:right].strip() if left < len(ln) else ""
        all_runs.append(row)

    # Compact view: only KEY_COLS that actually exist in this file
    present_keys = [k for k in KEY_COLS if k in col_names]
    compact = [{k: r.get(k, "") for k in present_keys} for r in all_runs]

    return {"runs": compact, "all_runs": all_runs, "n_runs": len(all_runs)}


def extract_plantgro_csv(job_dir: Path) -> Optional[Path]:
    """Convert PlantGro.OUT to trajectory.csv.

    Improvements over naive whitespace-split:
    - Prepends TRNO column (from *RUN lines) to distinguish runs in batch output.
    - Prepends DATE column (ISO 8601 YYYY-MM-DD) derived from YEAR + DOY.
    - Writes the column header only once even when multiple runs share the file.
    """
    src = job_dir / "PlantGro.OUT"
    if not src.exists():
        return None
    dst = job_dir / "trajectory.csv"

    def _doy_to_date(year: int, doy: int) -> str:
        try:
            return (dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)).isoformat()
        except Exception:
            return ""

    with src.open("r", errors="ignore") as f_in, dst.open("w", newline="") as f_out:
        w = csv.writer(f_out)
        header_written = False
        current_trno = 0
        for line in f_in:
            stripped = line.strip()
            if not stripped:
                continue
            # Track treatment number from "*RUN   1   : ..." lines
            if stripped.startswith("*RUN"):
                try:
                    current_trno = int(stripped.split()[1])
                except (IndexError, ValueError):
                    current_trno += 1
                continue
            # Column header line (repeated once per run block in multi-run output)
            if stripped.startswith("@") and "YEAR" in stripped:
                if not header_written:
                    w.writerow(["TRNO", "DATE"] + stripped.lstrip("@ ").split())
                    header_written = True
                continue  # skip duplicate headers from subsequent runs
            # Data row: starts with a 4-digit year
            if stripped and stripped[0].isdigit():
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        date_str = _doy_to_date(int(parts[0]), int(parts[1]))
                    except (ValueError, IndexError):
                        date_str = ""
                    w.writerow([current_trno, date_str] + parts)
    return dst


def _extract_daily_out_csv(job_dir: Path, src_name: str, dst_name: str) -> Optional[Path]:
    """Generic daily DSSAT OUT file → CSV converter.

    Works for SoilWat.OUT, SoilNi.OUT, ET.OUT, etc.
    All share the same format: *RUN header lines + @YEAR DOY ... column header + data rows.
    Prepends TRNO and DATE (ISO) columns identical to extract_plantgro_csv.
    """
    src = job_dir / src_name
    if not src.exists():
        return None
    dst = job_dir / dst_name
    with src.open("r", errors="ignore") as f_in, dst.open("w", newline="") as f_out:
        w = csv.writer(f_out)
        header_written = False
        current_trno = 0
        for line in f_in:
            stripped = line.strip()
            if not stripped or stripped.startswith("!"):
                continue
            if stripped.startswith("*RUN"):
                try:
                    current_trno = int(stripped.split()[1])
                except (IndexError, ValueError):
                    current_trno += 1
                continue
            if stripped.startswith("@YEAR") and "DOY" in stripped:
                if not header_written:
                    w.writerow(["TRNO", "DATE"] + stripped.lstrip("@").split())
                    header_written = True
                continue
            if stripped and stripped[0].isdigit():
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        date_str = (
                            dt.date(int(parts[0]), 1, 1) + dt.timedelta(days=int(parts[1]) - 1)
                        ).isoformat()
                    except (ValueError, IndexError):
                        date_str = ""
                    w.writerow([current_trno, date_str] + parts)
    return dst


def extract_soilwat_csv(job_dir: Path) -> Optional[Path]:
    """Convert SoilWat.OUT → soilwat.csv (daily soil water by layer)."""
    return _extract_daily_out_csv(job_dir, "SoilWat.OUT", "soilwat.csv")


def extract_soilni_csv(job_dir: Path) -> Optional[Path]:
    """Convert SoilNi.OUT → soilni.csv (daily soil NO3/NH4 by layer)."""
    return _extract_daily_out_csv(job_dir, "SoilNi.OUT", "soilni.csv")


def parse_overview_out(job_dir: Path) -> Optional[Dict[str, Any]]:
    """Parse DSSAT OVERVIEW.OUT for phenology and growth stage summary.

    OVERVIEW.OUT contains run-by-run narrative output including:
    - *PHENOLOGY: growth stage dates and GDD accumulation
    - Yield and biomass summary per run

    Returns a dict with 'runs': list of {trno, phenology, yield_summary}.
    Returns None if OVERVIEW.OUT is not found.
    """
    src = job_dir / "OVERVIEW.OUT"
    if not src.exists():
        return None

    try:
        lines = src.read_text(errors="ignore").splitlines()
    except Exception:
        return None

    runs: List[Dict[str, Any]] = []
    current_run: Dict[str, Any] = {}
    in_phenology = False
    pheno_header: List[str] = []
    pheno_rows: List[Dict[str, str]] = []

    def _flush_run():
        nonlocal current_run, in_phenology, pheno_header, pheno_rows
        if current_run:
            if pheno_rows:
                current_run["phenology_table"] = pheno_rows
            runs.append(current_run)
        current_run = {}
        in_phenology = False
        pheno_header = []
        pheno_rows = []

    for ln in lines:
        stripped = ln.strip()

        # New run block
        if stripped.startswith("*RUN"):
            _flush_run()
            parts = stripped.split(":")
            trno_part = parts[0].replace("*RUN", "").strip()
            try:
                current_run["trno"] = int(trno_part.split()[0])
            except Exception:
                current_run["trno"] = len(runs) + 1
            current_run["run_header"] = stripped
            continue

        if not current_run:
            continue

        # Phenology section header
        if "*PHENOLOGY" in stripped.upper() or "PHENOLOGY" == stripped.upper().strip("* "):
            in_phenology = True
            pheno_rows = []
            pheno_header = []
            continue

        # Exit phenology on new section
        if in_phenology and stripped.startswith("*") and "PHENOLOGY" not in stripped.upper():
            in_phenology = False

        # Phenology column header line
        if in_phenology and stripped.startswith("GROWTH") and not pheno_header:
            # skip the narrative header, wait for the data header
            pass
        elif in_phenology and stripped.startswith("@"):
            pheno_header = stripped.lstrip("@ ").split()
            continue

        # Phenology data rows
        if in_phenology and pheno_header and stripped and not stripped.startswith("!"):
            parts = stripped.split()
            if parts and not parts[0].startswith("*"):
                row = {}
                for i, col in enumerate(pheno_header):
                    row[col] = parts[i] if i < len(parts) else ""
                pheno_rows.append(row)

        # Capture yield/biomass lines in OVERVIEW (look for key keywords)
        for keyword, key in [
            ("YIELD", "yield_line"),
            ("BIOMASS", "biomass_line"),
            ("GRAIN", "grain_line"),
        ]:
            if keyword in stripped.upper() and ":" in stripped:
                current_run.setdefault("summary_lines", []).append(stripped)

    _flush_run()

    return {"n_runs": len(runs), "runs": runs} if runs else None


def extract_overview_csv(job_dir: Path) -> Optional[Path]:
    """Extract OVERVIEW.OUT phenology table to overview_phenology.csv.

    Writes one row per growth stage per run, with TRNO prepended.
    Returns None if OVERVIEW.OUT is missing or has no phenology data.
    """
    overview = parse_overview_out(job_dir)
    if not overview:
        return None

    dst = job_dir / "overview_phenology.csv"
    rows: List[Dict[str, str]] = []
    for run in overview["runs"]:
        trno = run.get("trno", "")
        for pheno_row in run.get("phenology_table", []):
            rows.append({"TRNO": str(trno), **pheno_row})

    if not rows:
        return None

    fieldnames = list(rows[0].keys())
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return dst


def _has_valid_yield(runs: List[Dict]) -> bool:
    """Return True if at least one run has a non-missing HWAM (> 0)."""
    for r in runs:
        try:
            if float(r.get("HWAM", "-99")) > 0:
                return True
        except (ValueError, TypeError):
            pass
    return False


# -----------------------------
# MCP SERVER & TOOLS
# -----------------------------
app = FastMCP("dssat-mcp")


@app.tool
def list_models() -> Dict[str, Any]:
    """List supported crops, weather stations, and soil profiles, with usage guidance.

    Call this first to understand what crops and data are available.
    Returns crop metadata, station/soil counts, and Korean-specific usage tips.

    Typical workflow:
      1. list_models()           → see available crops and defaults
      2. list_stations("SUWO")   → find Suwon weather file
      3. list_soils("KR")        → find Korean soil profiles
      4. list_cultivars("maize") → choose a cultivar
      5. run_simulation(...)     → run the simulation
    """
    n_wth = len(list(DSSAT_WEATHER.glob("*.WTH"))) if DSSAT_WEATHER.exists() else 0
    n_sol = len(list(DSSAT_SOIL.glob("*.SOL"))) if DSSAT_SOIL.exists() else 0

    # Per-crop guidance metadata (Korean context)
    crop_guide = {
        "maize": {
            "korean_name": "옥수수",
            "typical_sowing_korea": "5월 상순~6월 초 (May 1 - Jun 10)",
            "typical_yield_korea_kg_ha": "4000-7000",
            "growing_season_days": "120-140",
            "default_weather": "SUWO2501",
            "default_soil": "KR_JD_MAI1",
            "default_cultivar": "KR0003 (Dacheongok)",
        },
        "wheat": {
            "korean_name": "밀",
            "typical_sowing_korea": "10월 하순~11월 중순 (Oct 25 - Nov 15)",
            "typical_yield_korea_kg_ha": "3000-5000",
            "growing_season_days": "200-220",
            "default_weather": "SUWO2501",
            "default_soil": "KR_JD_MAI2",
            "default_cultivar": "IB0488 (NEWTON)",
        },
        "rice": {
            "korean_name": "벼",
            "typical_sowing_korea": "5월 중순~6월 초 이앙 (transplant May 15 - Jun 5)",
            "typical_yield_korea_kg_ha": "5000-7000",
            "growing_season_days": "130-160",
            "default_weather": "SUWO2501",
            "default_soil": "KR_JD_MAI2",
            "default_cultivar": "IB0012 (IR 58)",
        },
        "soybean": {
            "korean_name": "콩",
            "typical_sowing_korea": "6월 중순~하순 (Jun 15 - Jun 30, after rainy season risk)",
            "typical_yield_korea_kg_ha": "1000-2000",
            "growing_season_days": "100-130",
            "default_weather": "SUWO2501",
            "default_soil": "KR_JD_MAI2",
            "default_cultivar": "KR2828 (KRUG2828)",
            "note": "SUWO2501의 6월 장마기 SRAD=0 데이터로 인해 6월 20일 이후 파종 권장",
        },
        "barley": {
            "korean_name": "보리",
            "typical_sowing_korea": "봄보리: 3월 초~중순 / 가을보리: 10월 하순~11월 초",
            "typical_yield_korea_kg_ha": "2500-4000",
            "growing_season_days": "90-120 (봄보리)",
            "default_weather": "SUWO2501",
            "default_soil": "KR_JD_MAI2",
            "default_cultivar": "KR0001 (탑골/Tapgol)",
        },
        "potato": {
            "korean_name": "감자",
            "typical_sowing_korea": "봄감자: 3월 하순~4월 초순 / 고랭지: 4월 하순~5월 초",
            "typical_yield_korea_kg_ha": "20000-40000 (tuber DM ~5000-10000)",
            "growing_season_days": "90-120",
            "default_weather": "SUWO2501",
            "default_soil": "KR_JD_MAI2",
            "default_cultivar": "IB0001 (MAJESTIC)",
            "note": "HWAM reports tuber DM yield. PLWT=1500 g/m2 seed tuber weight built-in.",
        },
        "sorghum": {
            "korean_name": "수수",
            "typical_sowing_korea": "5월 하순~6월 초 (late May - early June)",
            "typical_yield_korea_kg_ha": "3000-6000",
            "growing_season_days": "100-130",
            "default_weather": "SUWO2501",
            "default_soil": "KR_JD_MAI1",
            "default_cultivar": "IB0001 (RIO)",
        },
    }

    crops_out = {}
    for k, d in SUPPORTED_CROPS.items():
        entry = {f: v for f, v in d.items() if f != "cul_file"}
        entry.update(crop_guide.get(k, {}))
        crops_out[k] = entry

    return {
        "supported_crops": list(SUPPORTED_CROPS.keys()),
        "crops": crops_out,
        "weather": {
            "note": f"Use list_stations(search) to browse {n_wth} available .WTH files",
            "korean_default": "SUWO2501 (수원, 37.26°N 126.98°E, elev 33 m)",
            "example_search": "list_stations('SUWO') for Suwon, list_stations('AMES') for Ames Iowa",
        },
        "soils": {
            "note": f"Use list_soils(search) to browse profiles across {n_sol} .SOL files",
            "korean_defaults": "KR_JD_MAI1 (Suwon Jungdong, SiC), KR_JD_MAI2 (Suwon Jungdong, SiL)",
            "example_search": "list_soils('KR') for Korean profiles, list_soils('IB') for IBSNAT",
        },
        "env": {
            "DSSAT_HOME": str(DSSAT_HOME),
            "DSSAT_WORK": str(DSSAT_WORK),
        },
    }


@app.tool
def list_cultivars(crop: str) -> Dict[str, Any]:
    """List available cultivars for a crop from the DSSAT genotype (.CUL) file.

    Args:
        crop: Crop type — "maize", "wheat", "rice", "soybean", "barley", "potato", or "sorghum".

    Returns a list of {var_id, name} dicts and the current default cultivar.
    Use the var_id value as the cultivar_id argument in run_simulation or run_batch.

    Korean cultivars available:
      maize:   KR0003 (Dacheongok/다청옥)
      soybean: KR2828 (KRUG2828)
      barley:  KR0001 (탑골/Tapgol), KR0002 (서둔찰/Seodunchal)
      potato:  IB0001 (MAJESTIC), IB0002 (SEBAGO), IB0003 (Russet Burbank)
      sorghum: IB0001 (RIO), IB0002 (9188), IB0004 (MN1500)
    """
    if crop not in SUPPORTED_CROPS:
        return {"ok": False, "error": f"unsupported_crop: {crop}. Supported: {list(SUPPORTED_CROPS)}"}
    cul_path = DSSAT_HOME / "Genotype" / SUPPORTED_CROPS[crop]["cul_file"]
    if not cul_path.exists():
        return {"ok": False, "error": f"CUL file not found: {cul_path}"}
    cultivars = parse_cul_file(cul_path)
    return {
        "ok": True,
        "crop": crop,
        "default_cultivar": SUPPORTED_CROPS[crop]["ingeno"],
        "n_cultivars": len(cultivars),
        "cultivars": cultivars,
    }


@app.tool
def list_stations(search: str = "") -> Dict[str, Any]:
    """List available weather stations by scanning the DSSAT Weather directory.

    Scans all .WTH files in DSSAT_WEATHER and reads lat/lon/elev from each
    file header.  No pre-registration required — any file found here can be
    passed as the weather_station argument to run_simulation or run_batch.

    Args:
        search: Optional case-insensitive filter string applied to the filename.
                E.g., "SUWO" for Suwon, "AMES" for Ames Iowa, "TX" for Texas,
                "25" for 2025 data files.  Leave empty to list everything.

    Returns a list of {file, station, insi, lat, lon, elev} dicts.
    Use the 'station' value (= file stem without .WTH) as weather_station.
    """
    if not DSSAT_WEATHER.exists():
        return {"ok": False, "error": f"DSSAT_WEATHER directory not found: {DSSAT_WEATHER}"}

    stations = []
    pattern = search.upper() if search else ""
    for wth_file in sorted(DSSAT_WEATHER.glob("*.WTH")):
        if pattern and pattern not in wth_file.name.upper():
            continue
        hdr = parse_wth_header(wth_file, read_data_range=True)
        stations.append({
            "file":       wth_file.name,
            "station":    wth_file.stem,    # pass this as weather_station
            "insi":       hdr["insi"],
            "lat":        hdr["lat"],
            "lon":        hdr["lon"],
            "elev":       hdr["elev"],
            "data_start": hdr.get("data_start", ""),
            "data_end":   hdr.get("data_end",   ""),
        })

    return {
        "ok": True,
        "n_stations": len(stations),
        "search": search,
        "weather_dir": str(DSSAT_WEATHER),
        "stations": stations,
    }


@app.tool
def list_soils(search: str = "") -> Dict[str, Any]:
    """List available soil profiles by scanning all .SOL files in DSSAT_SOIL.

    Any profile_id found here can be passed as the soil_profile argument to
    run_simulation or run_batch — no pre-registration required.

    Args:
        search: Optional case-insensitive filter applied to profile_id or
                description.  E.g., "KR" for Korean soils, "IB" for IBSNAT
                profiles, "Norfolk" for a specific series name.

    Returns a list of {profile_id, description, sol_file} dicts.
    Use the 'profile_id' value as soil_profile.
    """
    if not DSSAT_SOIL.exists():
        return {"ok": False, "error": f"DSSAT_SOIL directory not found: {DSSAT_SOIL}"}

    profiles = []
    pattern = search.upper() if search else ""
    for sol_file in sorted(DSSAT_SOIL.glob("*.SOL")):
        try:
            for line in sol_file.read_text(errors="ignore").splitlines():
                s = line.strip()
                # Profile header lines: "*PROFILE_ID  description"
                # Exclude *RUN section markers (not profile IDs)
                if (
                    s.startswith("*")
                    and not s.startswith("*RUN")
                    and len(s) > 1
                    and s[1:2].isalnum()
                ):
                    rest = s[1:]  # strip leading *
                    parts = rest.split(None, 1)
                    profile_id = parts[0]
                    # Skip comment-like tokens (contain ":" or end with ":")
                    if ":" in profile_id:
                        continue
                    description = parts[1].strip() if len(parts) > 1 else ""
                    if pattern and (
                        pattern not in profile_id.upper()
                        and pattern not in description.upper()
                    ):
                        continue
                    profiles.append({
                        "profile_id":  profile_id,
                        "description": description,
                        "sol_file":    sol_file.name,
                    })
        except Exception:
            continue

    return {
        "ok": True,
        "n_profiles": len(profiles),
        "search": search,
        "soil_dir": str(DSSAT_SOIL),
        "profiles": profiles,
    }


@app.tool
def run_simulation(
    crop: str,
    sowing_date: str,
    soil_profile: str,
    nitrogen_kg_ha: float = 120.0,
    irrigation_mm: float = 0.0,
    weather_station: str = "SUWO2501",
    delta_temp: float = 0.0,
    rainfall_factor: float = 1.0,
    cultivar_id: Optional[str] = None,
    ppop: Optional[float] = None,
    plrs: Optional[float] = None,
    fert_dap: int = 14,
    irr_dap: int = 14,
    co2_ppm: Optional[float] = None,
    fertilizer_events: Optional[List[Dict[str, Any]]] = None,
    ic_wr: Optional[float] = None,
    irrigation_events: Optional[List[Dict[str, Any]]] = None,
    auto_irrigate: bool = False,
    irr_threshold: float = 50.0,
    irr_target: float = 100.0,
    irr_depth_cm: int = 30,
    irr_amount_mm: float = 30.0,
    irr_method: str = "IR001",
    tillage_events: Optional[List[Dict[str, Any]]] = None,
    residue_events: Optional[List[Dict[str, Any]]] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a DSSAT crop growth simulation for a single scenario.

    Args:
        crop: Crop type — "maize", "wheat", "rice", "soybean", "barley", "potato", or "sorghum".
        sowing_date: Planting date as YYYY-MM-DD, e.g. "2025-05-01".
        soil_profile: Soil profile ID from any .SOL file. Use list_soils() to browse options.
                      Korean defaults: "KR_JD_MAI1" or "KR_JD_MAI2" (Suwon Jungdong).
                      US example: "IBSB910026" (Ida Silt Loam, Iowa).
        nitrogen_kg_ha: Total N fertilizer as single basal application (kg N/ha). Default 120.
                        Ignored if fertilizer_events is provided.
        irrigation_mm: Supplemental irrigation applied once at irr_dap after sowing (mm). Default 0.
        weather_station: WTH file stem (without .WTH extension). Use list_stations() to browse.
                         Korean default: "SUWO2501" (수원, Suwon 2025).
                         Example: "SUWO2501", "AMES9601" (Ames Iowa 1996).
        delta_temp: Temperature offset for climate scenario (°C). E.g. 2.0 = +2 °C warming.
        rainfall_factor: Rainfall scaling multiplier. E.g. 0.8 = 20 % less rain. Default 1.0.
        cultivar_id: Cultivar code from list_cultivars(). If omitted, crop default is used.
                     Korean cultivars: KR0003 (maize), KR2828 (soybean), KR0001/KR0002 (barley).
        ppop: Plant population (plants/m²). Uses crop default if omitted.
        plrs: Row spacing (cm). Uses crop default if omitted.
        fert_dap: Days after planting to apply basal N fertilizer (single event). Default 14.
        irr_dap:  Days after planting to apply supplemental irrigation. Default 14.
        co2_ppm:  Atmospheric CO2 concentration (ppm) for climate change scenario. Default None
                  (uses DSSAT default ~380-425 ppm depending on simulation year).
                  Examples: 450 (current), 550 (RCP4.5 mid-century), 700 (RCP8.5 end-century).
        fertilizer_events: List of fertilizer application events for split-N management.
                  Each dict must contain: dap (days after planting, int), n_kg_ha (float).
                  Optional keys: p_kg_ha (phosphorus), k_kg_ha (potassium).
                  Example (split N): [{"dap": 14, "n_kg_ha": 60}, {"dap": 45, "n_kg_ha": 60}]
                  If provided, overrides nitrogen_kg_ha + fert_dap.
        ic_wr: Initial soil water content as fraction of available water capacity (0–1).
               0 = wilting point, 1 = field capacity. Default 0.7 (70% AWC).
               Use lower values (0.3–0.5) for dry sowing; 1.0 for irrigated transplanting.
        irrigation_events: List of scheduled irrigation events.
               Each dict must contain: dap (days after planting, int), amount_mm (float).
               Optional key: irop (method code: "IR001"=sprinkler, "IR004"=drip). Default IR001.
               Example: [{"dap": 14, "amount_mm": 50}, {"dap": 45, "amount_mm": 50}]
               Overrides irrigation_mm + irr_dap when provided.
        auto_irrigate: If True, DSSAT automatically applies irrigation whenever soil water
               drops below irr_threshold (IRRIG=A mode). Overrides irrigation_events.
               Useful for potential yield estimation under no water stress.
        irr_threshold: % AWC at which auto-irrigation triggers (default 50). Used with auto_irrigate.
        irr_target: % AWC to refill to after auto-irrigation (default 100). Used with auto_irrigate.
        irr_depth_cm: Soil depth (cm) to monitor for auto-irrigation trigger (default 30).
        irr_amount_mm: Amount applied per auto-irrigation event, mm (default 30).
        irr_method: DSSAT irrigation method code (default "IR001"=sprinkler, "IR004"=drip).
        tillage_events: List of tillage operations. Each dict must contain:
               dap (int, days after planting; negative = before planting),
               implement (str, DSSAT code; default "TI009" = tandem disk).
               Optional keys: depth_cm (float, cm; default = implement default),
               name (str, label).
               Common codes: TI001=V-Ripper, TI003=Moldboard 20cm, TI004=Chisel sweeps,
               TI007=Disk plow, TI009=Tandem disk, TI014=Spike harrow.
               Example: [{"dap": -7, "implement": "TI007", "depth_cm": 25}]
        residue_events: List of crop residue / organic fertilizer applications.
               Each dict must contain: dap (int, days after planting; negative = before),
               amount_kg_ha (float, dry matter in kg/ha).
               Optional keys: rcod (residue code; RE001=maize, RE002=wheat, RE003=rice,
               RF001=farmyard manure, RF002=poultry manure), resn_pct (N%), resp_pct (P%),
               resk_pct (K%), incorporation_pct (0=surface mulch, 100=fully incorporated),
               depth_cm (incorporation depth cm).
               Example: [{"dap": -7, "rcod": "RE002", "amount_kg_ha": 3000, "incorporation_pct": 0}]
        job_id: Optional job identifier (reuse with get_result). Auto-generated if omitted.

    Returns a dict with:
        ok: True if successful
        human_summary: Human-readable result (ISO dates, labeled yield/biomass/N values)
        summary: Raw DSSAT output columns (HWAM, CWAM, PDAT, EDAT, ADAT, MDAT, PRCM, ETCM, NICM…)
        return_code: DSSAT exit code (0 = success)
        error_detail: DSSAT error message (only present if return_code != 0)
        job_id: Use with get_result() to retrieve this result later
    """
    if crop not in SUPPORTED_CROPS:
        return {"ok": False, "error": f"unsupported_crop: {crop}. Supported: {list(SUPPORTED_CROPS)}"}

    # Validate cultivar_id early if provided
    resolved_ingeno, resolved_cname = None, None
    if cultivar_id:
        try:
            resolved_ingeno, resolved_cname = lookup_cultivar(crop, cultivar_id)
        except (ValueError, FileNotFoundError) as e:
            return {"ok": False, "error": str(e)}

    try:
        spec = RunPayload(
            crop=crop,
            sowing_date=sowing_date,
            soil_profile=soil_profile,
            management=ManagementSpec(
                nitrogen_fertilizer_kg_ha=nitrogen_kg_ha,
                irrigation_mm=irrigation_mm,
            ),
            climate_scenario=ClimateScenario(
                weather=weather_station,
                delta_temp=delta_temp,
                rainfall_factor=rainfall_factor,
            ),
            job_id=job_id,
        )
    except ValidationError as e:
        return {"ok": False, "error": "invalid_payload", "details": json.loads(e.json())}

    job_id = spec.job_id or uuid.uuid4().hex[:12]
    job_dir = DSSAT_WORK / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    perturbed_wth_files: List[Path] = []  # perturbed files written to DSSAT_WEATHER; cleaned up after run
    wth_warnings: List[str] = []

    try:
        ensure_soil(spec.soil_profile, job_dir)  # copy soil file; shared across all scenarios

        # Check WTH data range before running
        sow_date = dt.date.fromisoformat(spec.sowing_date)
        crop_hlast = SUPPORTED_CROPS[spec.crop].get("hlast_days", 200)
        wth_warn = check_wth_range(spec.climate_scenario.weather, sow_date, crop_hlast)
        if wth_warn:
            wth_warnings.append(f"[{spec.climate_scenario.weather}] {wth_warn}")

        filex_list: List[Path] = []
        root_wth, root_wsta, wth_lat, wth_lon, wth_elev = prepare_weather(
            spec.climate_scenario.weather, spec.climate_scenario, job_dir, trno=1
        )
        if abs(spec.climate_scenario.delta_temp) > 1e-9 or abs(spec.climate_scenario.rainfall_factor - 1.0) > 1e-9:
            perturbed_wth_files.append(root_wth)
        filex_list.append(write_filex(
            spec, job_dir, trno=1, wsta=root_wsta,
            cultivar_id=resolved_ingeno, cname=resolved_cname,
            ppop_override=ppop, plrs_override=plrs,
            fert_dap=fert_dap, irr_dap=irr_dap,
            lat=wth_lat, lon=wth_lon, elev=wth_elev,
            co2_ppm=co2_ppm, fertilizer_events=fertilizer_events,
            ic_wr=ic_wr,
            irrigation_events=irrigation_events,
            auto_irrigate=auto_irrigate,
            irr_threshold=irr_threshold, irr_target=irr_target,
            irr_depth_cm=irr_depth_cm, irr_amount_mm=irr_amount_mm,
            irr_method=irr_method,
            tillage_events=tillage_events,
            residue_events=residue_events,
        ))

        for i, bi in enumerate(spec.batch, start=2):
            merged = RunPayload(
                crop=spec.crop,
                sowing_date=bi.sowing_date or spec.sowing_date,
                soil_profile=spec.soil_profile,
                management=bi.management or spec.management,
                climate_scenario=bi.climate_scenario or spec.climate_scenario,
                controls=spec.controls,
            )
            batch_wth, batch_wsta, b_lat, b_lon, b_elev = prepare_weather(
                merged.climate_scenario.weather, merged.climate_scenario, job_dir, trno=i
            )
            if abs(merged.climate_scenario.delta_temp) > 1e-9 or abs(merged.climate_scenario.rainfall_factor - 1.0) > 1e-9:
                perturbed_wth_files.append(batch_wth)
            filex_list.append(write_filex(
                merged, job_dir, trno=i, wsta=batch_wsta,
                lat=b_lat, lon=b_lon, elev=b_elev,
            ))

        # Create batch file
        write_dssbatch(filex_list, job_dir, crop_name=spec.crop)
    except (ValueError, FileNotFoundError) as exc:
        return {"ok": False, "error": type(exc).__name__, "details": str(exc)}

    # Try to link DSCSM048.CTR into job dir for convenience if present next to binary
    ctr_src = DSSAT_BIN.parent / "DSCSM048.CTR"
    if ctr_src.exists():
        shutil.copy2(ctr_src, job_dir / ctr_src.name)

    # Execute DSSAT
    model_code = SUPPORTED_CROPS[spec.crop]["model"]
    proc = run_dssat(job_dir, model_code)

    # Clean up perturbed weather files written to DSSAT_WEATHER
    for wth in perturbed_wth_files:
        try:
            wth.unlink()
        except Exception:
            pass

    # Parse outputs
    summary      = parse_summary_out(job_dir)
    traj_csv     = extract_plantgro_csv(job_dir)
    soilwat_csv  = extract_soilwat_csv(job_dir)
    soilni_csv   = extract_soilni_csv(job_dir)
    overview_csv = extract_overview_csv(job_dir)

    # Build human-readable summary from first run row
    human_summary = None
    runs = summary.get("runs", [])
    if runs:
        human_summary = _make_human_summary(runs[0], spec.crop)

    # On failure, include ERROR.OUT for diagnosis
    error_detail = None
    if proc.returncode != 0:
        err_file = job_dir / "ERROR.OUT"
        if err_file.exists():
            error_detail = err_file.read_text(errors="replace")[-3000:]

    # Write manifest for reproducibility
    write_manifest(job_dir, {
        "job_id": job_id,
        "payload": json.loads(spec.model_dump_json()),
        "return_code": proc.returncode,
        "stderr": proc.stderr[-4000:],
    })

    result: Dict[str, Any] = {
        "ok": proc.returncode == 0 and bool(runs) and _has_valid_yield(runs),
        "job_id": job_id,
        "return_code": proc.returncode,
        "human_summary": human_summary,
        "summary": summary,
        "trajectory_csv":    str(traj_csv)     if traj_csv     else None,
        "soilwat_csv":       str(soilwat_csv)  if soilwat_csv  else None,
        "soilni_csv":        str(soilni_csv)   if soilni_csv   else None,
        "overview_csv":      str(overview_csv) if overview_csv else None,
        "manifest_path": str(job_dir / "manifest.json"),
    }
    if error_detail:
        result["error_detail"] = error_detail
    if wth_warnings:
        result["warnings"] = wth_warnings
    return result


@app.tool
def run_batch(
    crop: str,
    soil_profile: str,
    scenarios: List[Dict[str, Any]],
    weather_station: str = "SUWO2501",
    co2_ppm: Optional[float] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run multiple DSSAT scenarios in a single batch job.

    Use this to compare sowing dates, climate scenarios, cultivars, or N rates
    in one call. crop and soil_profile are shared; each scenario can override
    any run_simulation parameter via its dict.

    Args:
        crop: Crop type — "maize", "wheat", "rice", "soybean", "barley", "potato", or "sorghum".
        soil_profile: Soil profile ID shared by all scenarios (e.g. "KR_JD_MAI1").
        scenarios: List of scenario dicts. Each dict can contain any subset of:
            sowing_date, nitrogen_kg_ha, irrigation_mm, delta_temp, rainfall_factor,
            cultivar_id, ppop, plrs, fert_dap, irr_dap, weather_station, label,
            co2_ppm (per-scenario CO2 override), fertilizer_events (list of N events),
            tillage_events (list of tillage operations, see run_simulation for format),
            residue_events (list of residue/organic fertilizer applications, see run_simulation).
            label: Optional string to name this scenario in results.
        weather_station: Default WTH file stem for all scenarios (overridable per scenario).
        co2_ppm: Default CO2 concentration (ppm) for all scenarios. Can be overridden per scenario.
                 E.g. 550 for RCP4.5, 700 for RCP8.5. Default None = model default (~425 ppm).
        job_id: Optional job identifier. Auto-generated if omitted.

    Per-scenario weather_station and co2_ppm overrides enable multi-year, multi-location,
    or climate scenario analysis. Returns a consolidated summary table with one row per scenario,
    each including human_summary (ISO dates, grain yield, biomass) and the scenario label.

    Example (climate change: CO2 + temperature):
        run_batch(
            crop="maize", soil_profile="KR_JD_MAI1",
            scenarios=[
                {"label": "baseline",      "sowing_date": "2025-05-01"},
                {"label": "+2C",           "sowing_date": "2025-05-01", "delta_temp": 2},
                {"label": "RCP4.5",        "sowing_date": "2025-05-01", "delta_temp": 2, "co2_ppm": 550},
                {"label": "RCP8.5",        "sowing_date": "2025-05-01", "delta_temp": 4, "co2_ppm": 700},
            ]
        )
    Example (split N fertilization):
        run_batch(
            crop="maize", soil_profile="KR_JD_MAI1",
            scenarios=[
                {"label": "single_120",    "sowing_date": "2025-05-01", "nitrogen_kg_ha": 120},
                {"label": "split_60+60",   "sowing_date": "2025-05-01",
                 "fertilizer_events": [{"dap": 14, "n_kg_ha": 60}, {"dap": 45, "n_kg_ha": 60}]},
                {"label": "split_40+80",   "sowing_date": "2025-05-01",
                 "fertilizer_events": [{"dap": 14, "n_kg_ha": 40}, {"dap": 45, "n_kg_ha": 80}]},
            ]
        )
    Example (multi-year with different WTH files):
        run_batch(
            crop="maize", soil_profile="IBSB910026",
            weather_station="AMES9601",
            scenarios=[
                {"label": "1990", "sowing_date": "1990-06-01", "weather_station": "AMES9001"},
                {"label": "1993", "sowing_date": "1993-06-01", "weather_station": "AMES9301"},
                {"label": "1996", "sowing_date": "1996-06-01", "weather_station": "AMES9601"},
            ]
        )
    """
    if not scenarios:
        return {"ok": False, "error": "scenarios list is empty"}
    if crop not in SUPPORTED_CROPS:
        return {"ok": False, "error": f"unsupported_crop: {crop}. Supported: {list(SUPPORTED_CROPS)}"}

    job_id = job_id or uuid.uuid4().hex[:12]
    job_dir = DSSAT_WORK / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    perturbed_wth_files: List[Path] = []
    filex_list: List[Path] = []
    scenario_meta: List[Dict] = []
    wth_warnings: List[str] = []

    try:
        ensure_soil(soil_profile, job_dir)

        crop_hlast = SUPPORTED_CROPS[crop].get("hlast_days", 200)

        for trno, sc in enumerate(scenarios, start=1):
            sowing_date   = sc.get("sowing_date",   "2025-05-01")
            n_kg          = float(sc.get("nitrogen_kg_ha", 120.0))
            irr_mm        = float(sc.get("irrigation_mm",  0.0))
            d_temp        = float(sc.get("delta_temp",     0.0))
            rf            = float(sc.get("rainfall_factor",1.0))
            sc_cultivar   = sc.get("cultivar_id")
            sc_ppop       = sc.get("ppop")
            sc_plrs       = sc.get("plrs")
            sc_fert_dap   = int(sc.get("fert_dap", 14))
            sc_irr_dap    = int(sc.get("irr_dap",  14))
            # Per-scenario weather_station overrides the top-level default
            sc_wstation   = sc.get("weather_station", weather_station)
            # Per-scenario CO2 override > top-level co2_ppm > None
            sc_co2        = sc.get("co2_ppm", co2_ppm)
            sc_fert_evts  = sc.get("fertilizer_events", None)
            sc_ic_wr      = sc.get("ic_wr", None)
            # Irrigation management
            sc_irr_evts   = sc.get("irrigation_events", None)
            sc_auto_irr   = bool(sc.get("auto_irrigate", False))
            sc_irr_thr    = float(sc.get("irr_threshold", 50.0))
            sc_irr_tgt    = float(sc.get("irr_target", 100.0))
            sc_irr_dep    = int(sc.get("irr_depth_cm", 30))
            sc_irr_amt    = float(sc.get("irr_amount_mm", 30.0))
            sc_irr_meth   = sc.get("irr_method", "IR001")
            sc_till_evts  = sc.get("tillage_events", None)
            sc_res_evts   = sc.get("residue_events", None)

            # Check WTH data range for this scenario
            sc_label = sc.get("label", f"S{trno:02d}")
            sow_date = dt.date.fromisoformat(sowing_date)
            wth_warn = check_wth_range(sc_wstation, sow_date, crop_hlast)
            if wth_warn:
                wth_warnings.append(f"[{sc_label}/{sc_wstation}] {wth_warn}")

            # Resolve cultivar if overridden
            sc_ingeno, sc_cname = None, None
            if sc_cultivar:
                try:
                    sc_ingeno, sc_cname = lookup_cultivar(crop, sc_cultivar)
                except Exception as e:
                    return {"ok": False, "error": f"Scenario {trno}: {e}"}

            climate = ClimateScenario(
                weather=sc_wstation, delta_temp=d_temp, rainfall_factor=rf
            )
            wth, wsta, wth_lat, wth_lon, wth_elev = prepare_weather(
                sc_wstation, climate, job_dir, trno=trno
            )
            if abs(d_temp) > 1e-9 or abs(rf - 1.0) > 1e-9:
                perturbed_wth_files.append(wth)

            payload = RunPayload(
                crop=crop, sowing_date=sowing_date, soil_profile=soil_profile,
                management=ManagementSpec(
                    nitrogen_fertilizer_kg_ha=n_kg, irrigation_mm=irr_mm
                ),
                climate_scenario=climate,
            )
            filex_list.append(write_filex(
                payload, job_dir, trno=trno, wsta=wsta,
                cultivar_id=sc_ingeno, cname=sc_cname,
                ppop_override=sc_ppop, plrs_override=sc_plrs,
                fert_dap=sc_fert_dap, irr_dap=sc_irr_dap,
                lat=wth_lat, lon=wth_lon, elev=wth_elev,
                co2_ppm=sc_co2, fertilizer_events=sc_fert_evts,
                ic_wr=sc_ic_wr,
                irrigation_events=sc_irr_evts,
                auto_irrigate=sc_auto_irr,
                irr_threshold=sc_irr_thr, irr_target=sc_irr_tgt,
                irr_depth_cm=sc_irr_dep, irr_amount_mm=sc_irr_amt,
                irr_method=sc_irr_meth,
                tillage_events=sc_till_evts,
                residue_events=sc_res_evts,
            ))
            scenario_meta.append({
                "trno": trno,
                "label": sc.get("label", f"S{trno:02d}"),
                **{k: v for k, v in sc.items() if k != "label"},
            })

        write_dssbatch(filex_list, job_dir, crop_name=crop)
    except (ValueError, FileNotFoundError, ValidationError) as exc:
        return {"ok": False, "error": type(exc).__name__, "details": str(exc)}

    ctr_src = DSSAT_BIN.parent / "DSCSM048.CTR"
    if ctr_src.exists():
        shutil.copy2(ctr_src, job_dir / ctr_src.name)

    model_code = SUPPORTED_CROPS[crop]["model"]
    proc = run_dssat(job_dir, model_code)

    for wth in perturbed_wth_files:
        try:
            wth.unlink()
        except Exception:
            pass

    summary      = parse_summary_out(job_dir)
    traj_csv     = extract_plantgro_csv(job_dir)
    soilwat_csv  = extract_soilwat_csv(job_dir)
    soilni_csv   = extract_soilni_csv(job_dir)
    overview_csv = extract_overview_csv(job_dir)

    # Attach scenario labels + human_summary to each run row.
    # In DSSAT batch mode each FileX is a separate experiment (TRNO=1 each);
    # RUNNO increments (1, 2, 3, …) across FileX files and is the correct index.
    runs = summary.get("runs", [])
    for run_row in runs:
        try:
            meta = scenario_meta[int(run_row.get("RUNNO", "1")) - 1]
            run_row["label"] = meta.get("label", "")
        except (ValueError, IndexError):
            pass
        run_row["human_summary"] = _make_human_summary(run_row, crop)

    # Build scenario comparison table (sorted by grain yield descending)
    comparison_table: List[Dict[str, Any]] = []
    for run_row in runs:
        hs = run_row.get("human_summary", {})
        comparison_table.append({
            "label":                         run_row.get("label", ""),
            "sowing_date":                   hs.get("sowing_date"),
            "maturity_date":                 hs.get("maturity_date"),
            "growing_days":                  hs.get("growing_days"),
            "grain_yield_kg_ha":             hs.get("grain_yield_kg_ha"),
            "total_biomass_kg_ha":           hs.get("total_biomass_kg_ha"),
            "harvest_index":                 hs.get("harvest_index"),
            "evapotranspiration_mm":         hs.get("evapotranspiration_mm"),
            "irrigation_applied_mm":         hs.get("irrigation_applied_mm"),
            "water_use_efficiency_kg_mm":    hs.get("water_use_efficiency_kg_mm"),
            "n_uptake_kg_ha":                hs.get("n_uptake_kg_ha"),
            "n_use_efficiency_kg_kg":        hs.get("n_use_efficiency_kg_kg"),
        })
    # Sort by grain yield descending (None values go last)
    comparison_table.sort(
        key=lambda r: float(r["grain_yield_kg_ha"]) if r["grain_yield_kg_ha"] is not None else -1,
        reverse=True,
    )
    # Add rank
    for rank, row in enumerate(comparison_table, start=1):
        row["rank"] = rank

    # On failure, include ERROR.OUT for diagnosis
    error_detail = None
    if proc.returncode != 0:
        err_file = job_dir / "ERROR.OUT"
        if err_file.exists():
            error_detail = err_file.read_text(errors="replace")[-3000:]

    write_manifest(job_dir, {
        "job_id": job_id, "scenarios": scenario_meta,
        "return_code": proc.returncode, "stderr": proc.stderr[-4000:],
    })

    result: Dict[str, Any] = {
        "ok": proc.returncode == 0 and bool(runs) and _has_valid_yield(runs),
        "job_id": job_id,
        "return_code": proc.returncode,
        "n_scenarios": len(scenarios),
        "summary": summary,
        "comparison_table": comparison_table,
        "trajectory_csv":    str(traj_csv)     if traj_csv     else None,
        "soilwat_csv":       str(soilwat_csv)  if soilwat_csv  else None,
        "soilni_csv":        str(soilni_csv)   if soilni_csv   else None,
        "overview_csv":      str(overview_csv) if overview_csv else None,
        "manifest_path": str(job_dir / "manifest.json"),
    }
    if error_detail:
        result["error_detail"] = error_detail
    if wth_warnings:
        result["warnings"] = wth_warnings
    return result


@app.tool
def get_result(job_id: str) -> Dict[str, Any]:
    """Fetch summary and file pointers for a previously run simulation job.

    Args:
        job_id: The job_id returned by run_simulation or run_batch.

    Returns the parsed Summary.OUT results, path to trajectory CSV (daily growth data),
    and the job manifest (inputs used). Useful to retrieve results without re-running.
    """
    job_dir = DSSAT_WORK / job_id
    if not job_dir.exists():
        return {"ok": False, "error": f"job_not_found: {job_id}"}
    summary = parse_summary_out(job_dir)
    traj = job_dir / "trajectory.csv"
    man = job_dir / "manifest.json"
    return {
        "ok": True,
        "job_id": job_id,
        "summary": summary,
        "trajectory_csv": str(traj) if traj.exists() else None,
        "manifest": json.loads(man.read_text()) if man.exists() else None,
        "job_dir": str(job_dir),
    }


@app.tool
def create_weather_station(
    location: str,
    start_date: str,
    end_date: str,
    kma_data_key: str = "",
    kma_station_key: str = "",
    srad_factor: float = 1.8,
) -> Dict[str, Any]:
    """KMA ASOS 일별 기상 데이터를 다운로드하여 DSSAT WTH 파일을 생성합니다.

    생성된 파일은 DSSAT_WEATHER 폴더에 저장되며, 반환된 wth_stem을
    run_simulation / run_batch 의 weather_station 인자로 바로 사용할 수 있습니다.

    Args:
        location:        지명 문자열 (예: "Suwon", "Jeonju") 또는 "위도,경도" 문자열.
        start_date:      시작일 "YYYY-MM-DD".
        end_date:        종료일 "YYYY-MM-DD" (미래 입력 시 어제로 자동 조정).
        kma_data_key:    공공데이터포털 ASOS 일별 서비스 인증키.
                         미입력 시 환경변수 KMA_DATA_API_KEY 사용.
        kma_station_key: apihub.kma.go.kr 관측소 목록 인증키.
                         미입력 시 환경변수 KMA_STATION_API_KEY 사용.
        srad_factor:     일조시간 → 일사량 변환 계수 (기본 1.8 MJ/m²/h).

    Returns:
        {
          "ok": bool,
          "message": str,
          "wth_stem": str,        # run_simulation weather_station 인자로 사용
          "wth_stems": list[str], # 연도별 stem 목록 (다년 조회 시)
          "wth_files": list[str], # 생성된 WTH 파일 경로 목록
          "station": str,         # 선택된 관측소명
          "station_id": str,      # KMA 관측소 번호
          "lat": float,
          "lon": float,
          "elev": float,
          "n_days": int,
          "date_range": str,
        }
    """
    if not _WEATHER_UTILS_OK:
        return {
            "ok": False,
            "message": (
                "weather_utils.py를 불러올 수 없습니다. "
                "weather_utils.py가 같은 폴더에 있는지 확인하고, "
                "pip install requests pandas geopy python-dateutil 을 실행하세요."
            ),
        }

    # "위도,경도" 문자열을 튜플로 변환
    loc: Any = location
    if "," in location:
        parts = location.split(",", 1)
        try:
            loc = (float(parts[0].strip()), float(parts[1].strip()))
        except ValueError:
            pass

    return _kma_download(
        location=loc,
        start_date=start_date,
        end_date=end_date,
        kma_data_key=kma_data_key,
        kma_station_key=kma_station_key,
        output_dir=str(DSSAT_WEATHER),
        srad_factor=srad_factor,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MODEL EVALUATION METRICS
# ──────────────────────────────────────────────────────────────────────────────

def _calc_model_metrics(observed: List[float], simulated: List[float]) -> Dict[str, Any]:
    """Calculate standard crop model evaluation statistics.

    Metrics:
      n      : number of paired observations
      rmse   : Root Mean Square Error (same units as data)
      mae    : Mean Absolute Error
      mbe    : Mean Bias Error (positive = model over-estimates)
      rmse_pct: RMSE as % of observed mean (normalised RMSE)
      d_index: Willmott (1981) index of agreement [0–1, 1 = perfect]
      nse    : Nash-Sutcliffe Efficiency [-∞–1, 1 = perfect]
      r2     : Coefficient of determination (Pearson r²)
      r      : Pearson correlation coefficient
    """
    n = len(observed)
    if n == 0:
        return {"n": 0, "error": "No paired observations"}
    if n != len(simulated):
        return {"n": 0, "error": f"Length mismatch: obs={len(observed)}, sim={len(simulated)}"}
    if n < 2:
        return {"n": n, "error": "At least 2 observations required for statistics"}

    obs = [float(o) for o in observed]
    sim = [float(s) for s in simulated]

    obs_mean = sum(obs) / n
    sim_mean = sum(sim) / n

    errors    = [s - o for o, s in zip(obs, sim)]
    sq_errors = [e ** 2 for e in errors]
    abs_errors = [abs(e) for e in errors]

    rmse     = math.sqrt(sum(sq_errors) / n)
    mae      = sum(abs_errors) / n
    mbe      = sum(errors) / n
    rmse_pct = (rmse / obs_mean * 100.0) if abs(obs_mean) > 1e-9 else None

    # Willmott d-index
    denom = sum((abs(s - obs_mean) + abs(o - obs_mean)) ** 2 for o, s in zip(obs, sim))
    d_index = (1.0 - sum(sq_errors) / denom) if denom > 1e-9 else None

    # Nash-Sutcliffe Efficiency
    ss_tot = sum((o - obs_mean) ** 2 for o in obs)
    nse    = (1.0 - sum(sq_errors) / ss_tot) if ss_tot > 1e-9 else None

    # Pearson r and R²
    cov     = sum((o - obs_mean) * (s - sim_mean) for o, s in zip(obs, sim)) / n
    obs_std = math.sqrt(sum((o - obs_mean) ** 2 for o in obs) / n)
    sim_std = math.sqrt(sum((s - sim_mean) ** 2 for s in sim) / n)
    if obs_std > 1e-9 and sim_std > 1e-9:
        r  = cov / (obs_std * sim_std)
        r2 = r ** 2
    else:
        r, r2 = None, None

    return {
        "n":         n,
        "rmse":      round(rmse,  2),
        "mae":       round(mae,   2),
        "mbe":       round(mbe,   2),
        "rmse_pct":  round(rmse_pct, 1) if rmse_pct is not None else None,
        "d_index":   round(d_index, 4) if d_index is not None else None,
        "nse":       round(nse,   4) if nse  is not None else None,
        "r2":        round(r2,    4) if r2   is not None else None,
        "r":         round(r,     4) if r    is not None else None,
    }


@app.tool
def evaluate_simulation(
    pairs: List[Dict[str, Any]],
    variables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate model evaluation statistics for one or more variables.

    Compares observed vs simulated values across multiple site-years
    and returns RMSE, MAE, MBE, d-index, NSE, and R² per variable.

    Args:
        pairs: List of dicts, each with "observed" and "simulated" sub-dicts.
               Both sub-dicts use the same variable names as keys.
               Supported variable names (use any subset):
                 grain_yield_kg_ha, total_biomass_kg_ha, growing_days,
                 n_uptake_kg_ha, evapotranspiration_mm, harvest_index,
                 anthesis_doy, maturity_doy  (day-of-year integers for dates)
               Example:
               [
                 {"observed": {"grain_yield_kg_ha": 5000, "growing_days": 135},
                  "simulated": {"grain_yield_kg_ha": 4919, "growing_days": 130}},
                 {"observed": {"grain_yield_kg_ha": 6200, "growing_days": 140},
                  "simulated": {"grain_yield_kg_ha": 5850, "growing_days": 138}},
               ]
        variables: Which variables to evaluate. Default = all numeric keys
                   found in the first pair.

    Returns:
        {
          "ok": bool,
          "n_pairs": int,
          "metrics": {
            "grain_yield_kg_ha": {n, rmse, mae, mbe, rmse_pct, d_index, nse, r2, r},
            ...
          },
          "pairs_used": [...],   # echoes back the paired values used
        }

    Usage example for calibration paper:
        evaluate_simulation(pairs=[
            {"observed": {"grain_yield_kg_ha": 4800, "growing_days": 132},
             "simulated": {"grain_yield_kg_ha": 4919, "growing_days": 130}},
            {"observed": {"grain_yield_kg_ha": 5200, "growing_days": 138},
             "simulated": {"grain_yield_kg_ha": 5050, "growing_days": 136}},
        ])
    """
    if not pairs:
        return {"ok": False, "error": "pairs list is empty"}

    # Determine variables to evaluate
    all_keys: List[str] = variables or []
    if not all_keys:
        # Auto-detect numeric variables from first pair
        first = pairs[0]
        obs0 = first.get("observed", {})
        sim0 = first.get("simulated", {})
        for k in list(obs0.keys()) + list(sim0.keys()):
            if k not in all_keys:
                val = obs0.get(k, sim0.get(k))
                try:
                    float(val)
                    all_keys.append(k)
                except (TypeError, ValueError):
                    pass

    if not all_keys:
        return {"ok": False, "error": "No numeric variables found in pairs"}

    # Collect observed/simulated lists per variable
    obs_by_var: Dict[str, List[float]] = {v: [] for v in all_keys}
    sim_by_var: Dict[str, List[float]] = {v: [] for v in all_keys}
    pairs_used: List[Dict] = []

    for i, pair in enumerate(pairs):
        obs_d = pair.get("observed", {})
        sim_d = pair.get("simulated", {})
        pair_record: Dict[str, Any] = {"pair_index": i}
        for v in all_keys:
            o_val = obs_d.get(v)
            s_val = sim_d.get(v)
            try:
                obs_by_var[v].append(float(o_val))
                sim_by_var[v].append(float(s_val))
                pair_record[v] = {"observed": float(o_val), "simulated": float(s_val),
                                   "error": round(float(s_val) - float(o_val), 2)}
            except (TypeError, ValueError):
                pair_record[v] = "missing"
        pairs_used.append(pair_record)

    # Compute metrics per variable
    metrics: Dict[str, Any] = {}
    for v in all_keys:
        obs_v = obs_by_var[v]
        sim_v = sim_by_var[v]
        if len(obs_v) >= 2:
            metrics[v] = _calc_model_metrics(obs_v, sim_v)
        elif len(obs_v) == 1:
            err = sim_v[0] - obs_v[0]
            metrics[v] = {
                "n": 1, "mbe": round(err, 2), "mae": round(abs(err), 2),
                "note": "Only 1 pair — cannot compute RMSE/d-index/NSE/r2",
            }
        else:
            metrics[v] = {"n": 0, "note": "No valid pairs for this variable"}

    return {
        "ok": True,
        "n_pairs": len(pairs),
        "variables_evaluated": all_keys,
        "metrics": metrics,
        "pairs_used": pairs_used,
    }


@app.tool
def sensitivity_analysis(
    crop: str,
    soil_profile: str,
    sowing_date: str,
    parameter: str,
    values: List[float],
    weather_station: str = "SUWO2501",
    base_nitrogen_kg_ha: float = 120.0,
    base_co2_ppm: Optional[float] = None,
    cultivar_id: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a one-at-a-time sensitivity analysis by varying a single parameter.

    Internally calls run_batch with one scenario per value, using all other
    parameters at their baseline. Returns a ranked comparison table showing
    how grain yield, biomass, and WUE respond to changes in the parameter.

    Args:
        crop: Crop type (maize, wheat, rice, soybean, barley, potato, sorghum).
        soil_profile: Soil profile ID (e.g. "KR_JD_MAI1").
        sowing_date: Baseline sowing date "YYYY-MM-DD".
        parameter: The parameter to vary. Supported:
            "nitrogen_kg_ha"   — N fertilizer rate (kg N/ha)
            "delta_temp"       — temperature offset (°C)
            "rainfall_factor"  — rainfall multiplier (1.0 = baseline)
            "co2_ppm"          — atmospheric CO2 concentration (ppm)
            "irrigation_mm"    — single irrigation event (mm)
            "ppop"             — plant population (plants/m²)
        values: List of values to test for the parameter.
                Example: [0, 60, 120, 180, 240] for nitrogen_kg_ha
                         [0, 1, 2, 3, 4]       for delta_temp
                         [280, 380, 550, 700]   for co2_ppm
        weather_station: WTH file stem (default "SUWO2501").
        base_nitrogen_kg_ha: Baseline N rate used for non-nitrogen scenarios.
        base_co2_ppm: Baseline CO2 (None = model default ~425 ppm).
        cultivar_id: Cultivar override (None = crop default).
        job_id: Optional job identifier.

    Returns:
        run_batch result with labeled comparison_table sorted by grain yield.
        Each label shows "{parameter}={value}".

    Example:
        sensitivity_analysis(
            crop="maize", soil_profile="KR_JD_MAI1",
            sowing_date="2025-05-01",
            parameter="nitrogen_kg_ha",
            values=[0, 60, 120, 180, 240],
        )
    """
    SUPPORTED_PARAMS = {
        "nitrogen_kg_ha", "delta_temp", "rainfall_factor",
        "co2_ppm", "irrigation_mm", "ppop",
    }
    if parameter not in SUPPORTED_PARAMS:
        return {
            "ok": False,
            "error": f"Unsupported parameter '{parameter}'. "
                     f"Choose from: {sorted(SUPPORTED_PARAMS)}",
        }
    if not values:
        return {"ok": False, "error": "values list is empty"}

    scenarios: List[Dict[str, Any]] = []
    for v in values:
        sc: Dict[str, Any] = {
            "sowing_date":      sowing_date,
            "nitrogen_kg_ha":   base_nitrogen_kg_ha,
            "weather_station":  weather_station,
            "label":            f"{parameter}={v}",
        }
        if base_co2_ppm is not None:
            sc["co2_ppm"] = base_co2_ppm
        if cultivar_id is not None:
            sc["cultivar_id"] = cultivar_id
        # Override the target parameter
        sc[parameter] = v
        scenarios.append(sc)

    # Use run_batch's inner function to avoid MCP wrapping issues
    _batch_fn = getattr(run_batch, "fn", run_batch)
    result = _batch_fn(
        crop=crop,
        soil_profile=soil_profile,
        scenarios=scenarios,
        weather_station=weather_station,
        co2_ppm=base_co2_ppm,
        job_id=job_id,
    )
    # Tag results with sensitivity analysis metadata
    result["sensitivity_parameter"] = parameter
    result["sensitivity_values"] = values
    return result


# ──────────────────────────────────────────────────────────────────────────────
# CULTIVAR PARAMETER ESTIMATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

# GDD base/opt temperatures for P5 (grain fill), per crop
_TBASE_P5: Dict[str, Tuple[float, float]] = {
    "maize":   (8.0,  34.0),
    "wheat":   (0.0,  30.0),
    "barley":  (0.0,  30.0),
    "sorghum": (10.0, 38.0),
    "rice":    (8.0,  33.0),
    "soybean": (6.0,  30.0),
    "potato":  (7.0,  26.0),
}


def _read_wth_daily(wth_stem: str) -> List[Dict[str, Any]]:
    """Read all daily rows from a WTH file.

    Returns list of dicts: {date, tmax, tmin, srad, rain}.
    """
    src: Optional[Path] = None
    candidate = DSSAT_WEATHER / f"{wth_stem}.WTH"
    if candidate.exists():
        src = candidate
    else:
        key_upper = wth_stem.upper()
        for f in DSSAT_WEATHER.glob("*.WTH"):
            if f.stem.upper() == key_upper:
                src = f
                break
    if src is None:
        raise FileNotFoundError(f"{wth_stem}.WTH not found in {DSSAT_WEATHER}")

    rows: List[Dict[str, Any]] = []
    in_data = False
    for ln in src.read_text(errors="ignore").splitlines():
        if ln.strip().startswith("@DATE"):
            in_data = True
            continue
        if in_data and ln.strip() and ln[0].isdigit() and len(ln) >= 29:
            try:
                yyddd = ln[0:5]
                srad  = float(ln[5:11])
                tmax  = float(ln[11:17])
                tmin  = float(ln[17:23])
                rain  = float(ln[23:29])
                date  = dt.date.fromisoformat(_yyddd_to_iso(yyddd))
                rows.append({"date": date, "tmax": tmax, "tmin": tmin,
                             "srad": srad, "rain": rain})
            except Exception:
                pass
    return rows


def _calc_gdd(
    daily: List[Dict[str, Any]],
    start: dt.date,
    end: dt.date,
    tbase: float,
    topt: float = 999.0,
) -> float:
    """Accumulate GDD from start to end (inclusive).

    GDD/day = max(0, min(topt, (Tmax+Tmin)/2) - Tbase)
    """
    total = 0.0
    for row in daily:
        if start <= row["date"] <= end:
            tmean = (row["tmax"] + row["tmin"]) / 2.0
            total += max(0.0, min(topt, tmean) - tbase)
    return round(total, 1)


def _calc_vern_days(
    daily: List[Dict[str, Any]],
    start: dt.date,
    end: dt.date,
    tlo: float = 0.0,
    thi: float = 7.0,
) -> float:
    """Count days where mean temperature falls in [tlo, thi] (vernalization window)."""
    count = 0.0
    for row in daily:
        if start <= row["date"] <= end:
            tmean = (row["tmax"] + row["tmin"]) / 2.0
            if tlo <= tmean <= thi:
                count += 1.0
    return round(count, 1)


def _cul_col_right(header: str, pname: str) -> int:
    """Return right-edge column index for a parameter in a DSSAT CUL header line.

    In DSSAT .CUL files, 6-char data fields are right-aligned under their column
    names. The rightmost character of the name aligns with the rightmost character
    of the value field.  So: field = line[right-5 : right+1].
    """
    pos = header.rfind(pname)
    if pos < 0:
        return -1
    return pos + len(pname) - 1


def _get_cul_path(crop_name: str) -> Path:
    """Return absolute path to the .CUL file for a crop."""
    return DSSAT_HOME / "Genotype" / SUPPORTED_CROPS[crop_name]["cul_file"]


def _read_base_cul_params(crop_name: str, cultivar_id: str) -> Dict[str, Any]:
    """Read all numeric parameters for a cultivar from its .CUL file.

    Returns dict with param names as float values, plus:
      _header: header line string
      _line:   raw cultivar data line
      _params_list: ordered list of parameter column names
    """
    cul_path = _get_cul_path(crop_name)
    lines = cul_path.read_text(errors="ignore").splitlines()

    header = ""
    params_list: List[str] = []

    for line in lines:
        if line.strip().startswith("@VAR"):
            header = line
            # Extract param names: tokens after "ECO#"
            eco_pos = header.upper().find("ECO#")
            if eco_pos >= 0:
                rest = header[eco_pos + 4:]
                params_list = []
                i = 0
                while i < len(rest):
                    if rest[i].isalpha() or rest[i].isdigit():
                        j = i
                        while j < len(rest) and rest[j] not in (" ", "\t"):
                            j += 1
                        params_list.append(rest[i:j])
                        i = j
                    else:
                        i += 1

    for line in lines:
        if not line or not line[0].isalnum():
            continue
        if line[0:6].strip().upper() == cultivar_id.upper():
            result: Dict[str, Any] = {
                "_header": header,
                "_line": line,
                "_params_list": params_list,
            }
            for pname in params_list:
                right = _cul_col_right(header, pname)
                if right < 0:
                    result[pname] = None
                    continue
                try:
                    # Python slice is safe even if right+1 > len(line)
                    result[pname] = float(line[right - 5:right + 1].strip())
                except Exception:
                    result[pname] = None
            return result

    raise ValueError(f"Cultivar '{cultivar_id}' not found in {cul_path}")


def _write_temp_cultivar(
    crop_name: str,
    base_id: str,
    new_id: str,
    new_params: Dict[str, float],
) -> Path:
    """Insert a temporary cultivar line into the .CUL file.

    Copies the base cultivar line, replaces the VAR# and specified parameter
    values, then inserts it right after the base cultivar line.
    Returns the cul_path.
    """
    cul_path = _get_cul_path(crop_name)
    lines = cul_path.read_text(errors="ignore").splitlines()

    header = ""
    base_idx = -1
    base_line = ""

    for i, line in enumerate(lines):
        if line.strip().startswith("@VAR"):
            header = line
        elif line and line[0].isalnum():
            if line[0:6].strip().upper() == base_id.upper():
                base_idx = i
                base_line = line

    if base_idx < 0:
        raise ValueError(f"Base cultivar '{base_id}' not found in {cul_path}")

    # Build new line from base
    new_chars = list(base_line.ljust(max(len(base_line), 80)))

    # Replace VAR# (columns 0-5)
    for j, ch in enumerate(new_id.ljust(6)[:6]):
        new_chars[j] = ch

    # Replace VRNAME (columns 7-24) with "Estimated"
    name_str = "Estimated          "
    for j, ch in enumerate(name_str[:18]):
        if 7 + j < len(new_chars):
            new_chars[7 + j] = ch

    # Replace each parameter value in-place
    for pname, val in new_params.items():
        right = _cul_col_right(header, pname)
        if right < 0 or val is None:
            continue
        # 6-char right-aligned field
        if abs(val) < 10.0:
            val_str = f"{val:6.2f}"
        else:
            val_str = f"{val:6.1f}"
        start = right - 5
        for k, ch in enumerate(val_str[:6]):
            pos = start + k
            if 0 <= pos < len(new_chars):
                new_chars[pos] = ch

    new_line = "".join(new_chars).rstrip()
    lines.insert(base_idx + 1, new_line)
    cul_path.write_text("\n".join(lines) + "\n")
    return cul_path


def _remove_temp_cultivar(crop_name: str, new_id: str) -> None:
    """Remove a temporary cultivar line from the .CUL file."""
    cul_path = _get_cul_path(crop_name)
    lines = cul_path.read_text(errors="ignore").splitlines()
    new_lines = [
        ln for ln in lines
        if not (ln and ln[0].isalnum() and ln[0:6].strip().upper() == new_id.upper())
    ]
    cul_path.write_text("\n".join(new_lines) + "\n")


@app.tool
def estimate_cultivar_params(
    crop: str,
    sowing_date: str,
    soil_profile: str,
    weather_station: str = "SUWO2501",
    observations: Optional[Dict[str, Any]] = None,
    base_cultivar_id: Optional[str] = None,
    nitrogen_kg_ha: float = 120.0,
) -> Dict[str, Any]:
    """관측 데이터로 DSSAT 품종 모수를 정의 기반으로 직접 추정합니다.

    최적화 루프 없이 모수 정의 + 기온 데이터(WTH)로 직접 계산하고,
    계산 후 DSSAT 시뮬레이션으로 결과를 검증합니다.

    Args:
        crop: 작물명 ("maize", "wheat", "barley", "rice", "soybean", "potato", "sorghum")
        sowing_date: 파종일 "YYYY-MM-DD"
        soil_profile: 토양 프로파일 ID (예: "KR_JD_MAI1")
        weather_station: 기상파일 스템 (예: "SUWO2501")
        base_cultivar_id: 기준 품종 ID (미지정 시 작물 기본값 사용)
        nitrogen_kg_ha: 검증 시뮬레이션 질소 시비량 (kg N/ha). Default 120.
        observations: 관측 데이터 딕셔너리. 사용 가능한 키:
            heading_date  (str "YYYY-MM-DD"): 출수/출사/개화일 (P5 계산 기준)
            maturity_date (str "YYYY-MM-DD"): 성숙일 (P5 계산 필수)
            yield_kg_ha   (float): 수량 (kg/ha)
            tgw_g         (float): 천립중 (g) — G2/G3 계산
            kernels_per_ear (float): 이삭당 낟알수 (옥수수/밀/보리)
            ears_per_m2   (float): 단위면적당 이삭수 (/m²)
            biomass_kg_ha (float): 지상부 건물중 성숙기 (kg/ha)
            biomass_at_anthesis_kg_ha (float): 출수기 지상부 건물중 (kg/ha) — G1 계산에 필수

    Returns:
        {
          "ok": bool,
          "estimated_params": {param: value, ...},  # 추정된 모수
          "final_params": {param: value, ...},        # 추정+기준 혼합 최종값
          "estimation_notes": [...],                   # 각 모수 추정 방법 설명
          "base_cultivar_params": {param: value, ...}, # 기준 품종 원래 모수
          "verification": {...},                       # DSSAT 검증 시뮬레이션 결과
          "temp_cultivar_id": "ESTM01",
          "base_cultivar_id": str,
        }
    """
    if crop not in SUPPORTED_CROPS:
        return {"ok": False, "error": f"Unsupported crop: {crop}. "
                f"Choose from: {list(SUPPORTED_CROPS.keys())}"}

    if observations is None:
        observations = {}

    base_id = base_cultivar_id or SUPPORTED_CROPS[crop]["ingeno"]
    temp_id = "ESTM01"

    # 1) Read base cultivar parameters
    try:
        base_params = _read_base_cul_params(crop, base_id)
        params_list: List[str] = base_params.get("_params_list", [])
    except Exception as e:
        return {"ok": False, "error": f"Failed to read base cultivar '{base_id}': {e}"}

    # 2) Load WTH daily weather data
    try:
        daily = _read_wth_daily(weather_station)
    except Exception as e:
        return {"ok": False, "error": f"Failed to read weather '{weather_station}': {e}"}

    sow = dt.date.fromisoformat(sowing_date)
    tbase, topt = _TBASE_P5.get(crop, (8.0, 38.0))

    # 3) Parse observation dates
    obs_heading: Optional[dt.date] = None
    obs_maturity: Optional[dt.date] = None
    for key, attr in [("heading_date", "obs_heading"), ("maturity_date", "obs_maturity")]:
        if observations.get(key):
            try:
                d = dt.date.fromisoformat(observations[key])
                if attr == "obs_heading":
                    obs_heading = d
                else:
                    obs_maturity = d
            except Exception:
                pass

    obs_yield   = observations.get("yield_kg_ha")
    obs_tgw_g   = observations.get("tgw_g")
    obs_kpe     = observations.get("kernels_per_ear")
    obs_epm2    = observations.get("ears_per_m2")
    obs_biomass = observations.get("biomass_kg_ha")
    ppop        = SUPPORTED_CROPS[crop]["ppop"]

    # 4) Estimate parameters from definitions
    estimated: Dict[str, float] = {}
    notes: List[str] = []

    # ── P5: GDD from heading/anthesis to maturity ──────────────────────────
    if obs_heading and obs_maturity and obs_heading < obs_maturity:
        p5 = _calc_gdd(daily, obs_heading, obs_maturity, tbase, topt)
        estimated["P5"] = p5
        notes.append(
            f"P5={p5:.1f} °C·d: GDD(Tbase={tbase}°C, Topt={topt}°C) "
            f"{obs_heading} → {obs_maturity} 적산"
        )
    elif obs_maturity and not obs_heading:
        notes.append("P5: heading_date 없음 — maturity_date만으로는 계산 불가, 기준 품종 값 유지")
    else:
        notes.append("P5: heading_date/maturity_date 미제공 — 기준 품종 값 유지")

    # ── Crop-specific estimation ───────────────────────────────────────────
    if crop in ("wheat", "barley"):
        # P1V: vernalization requirement (days with mean T in 0–7 °C)
        # Active vernalization period: sowing to sowing+90 days
        vern_end = sow + dt.timedelta(days=90)
        p1v = _calc_vern_days(daily, sow, vern_end, tlo=0.0, thi=7.0)
        estimated["P1V"] = p1v
        notes.append(
            f"P1V={p1v:.1f} 일: 파종({sow})~파종+90일 중 "
            f"일평균기온 0–7°C 누적 일수 (춘화 요구 일수)"
        )

        # G2: kernel weight (mg) — numerically equal to TGW in grams
        if obs_tgw_g is not None:
            estimated["G2"] = float(obs_tgw_g)
            notes.append(
                f"G2={obs_tgw_g:.2f} mg: 천립중 {obs_tgw_g}g → "
                f"낟알 1개 무게 {obs_tgw_g}mg"
            )

        # G1: kernels per unit canopy weight at anthesis (kernels / g·m⁻²)
        # G1 정의: 출수기 단위 건물중당 낟알수.
        # 출수기 실측 건물중(biomass_at_anthesis_kg_ha)이 있을 때만 신뢰성 있게 계산 가능.
        # HI·건물중 가정으로 역산한 출수기 건물중은 오차가 크므로 G1 추정에 사용하지 않음.
        obs_bio_anth = observations.get("biomass_at_anthesis_kg_ha")
        if obs_kpe and obs_epm2 and obs_bio_anth:
            kernels_m2 = float(obs_kpe) * float(obs_epm2)
            bio_anth_g_m2 = float(obs_bio_anth) * 0.1   # kg/ha → g/m²
            if bio_anth_g_m2 > 0:
                g1 = kernels_m2 / bio_anth_g_m2
                estimated["G1"] = round(g1, 2)
                notes.append(
                    f"G1={g1:.2f}: {obs_kpe}낟알/이삭 × {obs_epm2}/m² "
                    f"÷ {bio_anth_g_m2:.1f} g/m² (출수기 실측 건물중)"
                )
        else:
            notes.append(
                "G1: 출수기 실측 건물중(biomass_at_anthesis_kg_ha) 없음 — "
                "기준 품종 값 유지 (HI 가정 역산은 오차 과대)"
            )

    elif crop == "maize":
        # G3: kernel weight (mg) = TGW (g) numerically
        if obs_tgw_g is not None:
            estimated["G3"] = float(obs_tgw_g)
            notes.append(
                f"G3={obs_tgw_g:.2f} mg: 천립중 {obs_tgw_g}g → "
                f"낟알 1개 무게 {obs_tgw_g}mg"
            )
        # G2: kernels per plant
        if obs_yield and obs_tgw_g:
            kernels_m2 = float(obs_yield) * 100.0 / float(obs_tgw_g)
            g2 = kernels_m2 / ppop
            estimated["G2"] = round(g2, 1)
            notes.append(
                f"G2={g2:.1f}: 낟알수 {kernels_m2:.0f}/m² "
                f"÷ {ppop} 식물체/m²"
            )
        elif obs_kpe:
            estimated["G2"] = float(obs_kpe)
            notes.append(f"G2={obs_kpe:.1f}: 이삭당 낟알수 직접 사용")

    elif crop == "sorghum":
        if obs_tgw_g is not None:
            estimated["G3"] = float(obs_tgw_g)
            notes.append(f"G3={obs_tgw_g:.2f} mg: 천립중 {obs_tgw_g}g")
        if obs_yield and obs_tgw_g:
            kernels_m2 = float(obs_yield) * 100.0 / float(obs_tgw_g)
            g2 = kernels_m2 / ppop
            estimated["G2"] = round(g2, 1)
            notes.append(f"G2={g2:.1f}: 낟알수 {kernels_m2:.0f}/m² ÷ {ppop}/m²")

    notes.append(
        "P1/P2/P1D/PHINT 등 나머지 모수: 관측값 없음 — 기준 품종 값 그대로 유지"
    )

    if not estimated:
        return {
            "ok": False,
            "error": (
                "추정 가능한 모수 없음. "
                "observations에 heading_date, maturity_date, tgw_g, yield_kg_ha 중 "
                "하나 이상을 제공하세요."
            ),
            "base_cultivar_params": {k: v for k, v in base_params.items()
                                     if not k.startswith("_")},
        }

    # 5) Write temporary cultivar into CUL file
    try:
        _write_temp_cultivar(crop, base_id, temp_id, estimated)
    except Exception as e:
        return {"ok": False, "error": f"CUL 파일 쓰기 실패: {e}"}

    # 6) Verification simulation with estimated parameters
    # run_simulation may be wrapped as FunctionTool by fastmcp; access .fn if so
    _run_fn = getattr(run_simulation, "fn", run_simulation)
    verification: Dict[str, Any] = {}
    try:
        sim = _run_fn(
            crop=crop,
            sowing_date=sowing_date,
            soil_profile=soil_profile,
            weather_station=weather_station,
            cultivar_id=temp_id,
            nitrogen_kg_ha=nitrogen_kg_ha,
        )
        hs = sim.get("human_summary", {})
        sim_mat  = hs.get("maturity_date")
        sim_yld  = hs.get("grain_yield_kg_ha")
        sim_days = hs.get("growing_days")
        sim_head = hs.get("anthesis_date")

        verification = {
            "ok": sim.get("ok"),
            "simulated_heading_date":      sim_head,
            "simulated_maturity_date":     sim_mat,
            "simulated_grain_yield_kg_ha": sim_yld,
            "simulated_growing_days":      sim_days,
        }

        # Errors vs observations
        if obs_maturity and sim_mat:
            try:
                err_d = (dt.date.fromisoformat(sim_mat) - obs_maturity).days
                verification["maturity_error_days"] = err_d
            except Exception:
                pass
        if obs_yield and sim_yld:
            try:
                err_pct = (float(sim_yld) - float(obs_yield)) / float(obs_yield) * 100.0
                verification["yield_error_pct"] = round(err_pct, 1)
            except Exception:
                pass

    except Exception as e:
        verification = {"ok": False, "error": str(e)}
    finally:
        try:
            _remove_temp_cultivar(crop, temp_id)
        except Exception:
            pass

    # 7) Build output
    base_out = {k: v for k, v in base_params.items() if not k.startswith("_")}
    final_params = {p: base_params.get(p) for p in params_list}
    final_params.update(estimated)

    return {
        "ok": verification.get("ok", False),
        "temp_cultivar_id": temp_id,
        "base_cultivar_id": base_id,
        "estimated_params": estimated,
        "final_params": final_params,
        "estimation_notes": notes,
        "base_cultivar_params": base_out,
        "verification": verification,
    }


if __name__ == "__main__":
    # Run MCP server (stdio). fastmcp handles the transport for MCP-compatible clients.
    app.run()
