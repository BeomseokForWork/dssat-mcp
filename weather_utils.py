"""
weather_utils.py — KMA ASOS 기상 데이터 다운로드 → DSSAT WTH 파일 생성

사용 방법:
    from weather_utils import create_weather_station

    result = create_weather_station(
        location="Suwon",
        start_date="2025-01-01",
        end_date="2025-12-31",
        kma_data_key="...",
        kma_station_key="...",
    )
    # result["wth_stem"] → "SUWO2501"  (run_simulation의 weather_station 인자로 사용)

환경 변수 (.env 또는 시스템):
    KMA_DATA_API_KEY      — 공공데이터포털 ASOS 일별 서비스 인증키
    KMA_STATION_API_KEY   — apihub.kma.go.kr 관측소 목록 인증키
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderUnavailable
    _GEOPY_OK = True
except ImportError:
    _GEOPY_OK = False

# ── 기본값 ────────────────────────────────────────────────────────────────────
_SRAD_FACTOR = 1.8          # 일조시간(h) → 일사량(MJ/m²/day): SRAD = SSHR × factor
_RATE_LIMIT   = 0.5         # API 요청 간격 (초)
_MAX_ELEV     = 3000.0      # 이 값 초과 시 표고 오류로 판단 → -99 처리


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_station_list(api_key: str) -> Optional[pd.DataFrame]:
    """apihub.kma.go.kr에서 지표면 관측소 목록을 가져와 DataFrame으로 반환."""
    url = "https://apihub.kma.go.kr/api/typ01/url/stn_inf.php"
    params = {
        "authKey": api_key,
        "inf": "SFC",
        "tm": datetime.now().strftime("%Y%m%d%H%M"),
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return None

    col_names = [
        "지점", "경도", "위도", "지점높이(m)", "HT_PA", "HT_TA", "HT_WD", "HT_RN",
        "지점명", "STN_EN", "FCT_ID", "LAW_ID", "BASIN",
    ]
    rows = []
    for line in resp.text.strip().splitlines():
        if line.strip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < len(col_names):
            continue
        stn_ko = " ".join(parts[8:-4])
        rows.append(parts[:8] + [stn_ko] + parts[-4:])

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=col_names)
    for col in ["위도", "경도", "지점높이(m)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["위도", "경도"], inplace=True)
    return df


def _fetch_asos_monthly(
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    station_id: str,
    api_key: str,
) -> List[dict]:
    """공공데이터포털 ASOS 일별 기상 데이터를 한 달치씩 반환."""
    url = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
    params = {
        "serviceKey": api_key,
        "pageNo": "1",
        "numOfRows": "999",
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": start_yyyymmdd,
        "endDt": end_yyyymmdd,
        "stnIds": station_id,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data["response"]["header"]["resultCode"] != "00":
            return []
        items = data["response"]["body"]["items"]["item"]
        return items if isinstance(items, list) else [items]
    except Exception:
        return []


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 좌표 사이의 거리 (km)."""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _nearest_station(lat: float, lon: float, df: pd.DataFrame):
    """stations_df에서 (lat, lon)에 가장 가까운 관측소를 반환."""
    dists = df.apply(
        lambda row: _haversine(lat, lon, row["위도"], row["경도"]), axis=1
    )
    idx = dists.idxmin()
    return df.loc[idx], dists[idx]


def _geocode(location: str) -> Tuple[Optional[float], Optional[float]]:
    """지명 → (위도, 경도). geopy 없으면 None, None."""
    if not _GEOPY_OK:
        return None, None
    try:
        geo = Nominatim(user_agent="dssat_mcp_weather")
        loc = geo.geocode(location)
        if loc:
            return loc.latitude, loc.longitude
        return None, None
    except Exception:
        return None, None


def _build_df(raw_items: List[dict], srad_factor: float) -> Optional[pd.DataFrame]:
    """API 응답 항목 리스트 → 정제된 DataFrame (DATE, TMAX, TMIN, RAIN, SRAD)."""
    if not raw_items:
        return None

    df = pd.DataFrame(raw_items)
    col_map = {"tm": "DATE", "maxTa": "TMAX", "minTa": "TMIN",
               "sumRn": "RAIN", "sumSsHr": "SSHR"}
    existing = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(existing.keys())].rename(columns=existing)

    for col in ["TMAX", "TMIN", "RAIN", "SSHR"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["DATE"] = pd.to_datetime(df["DATE"])
    df.fillna(-99.0, inplace=True)

    # 강수량 결측 → 0 (측정 안 된 날 = 강수 없음)
    if "RAIN" in df.columns:
        df.loc[df["RAIN"] == -99.0, "RAIN"] = 0.0

    # 일조시간 → 일사량 변환
    if "SSHR" in df.columns:
        df["SRAD"] = df["SSHR"].apply(
            lambda x: round(x * srad_factor, 1) if x >= 0 else -99.0
        )
    else:
        df["SRAD"] = -99.0

    # TMAX ≤ TMIN 보정 (DSSAT 치명 오류 방지)
    if "TMAX" in df.columns and "TMIN" in df.columns:
        mask = (df["TMIN"] >= df["TMAX"]) & (df["TMAX"] > -90)
        df.loc[mask, "TMIN"] = df.loc[mask, "TMAX"] - 1.0

    df.sort_values("DATE", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _write_wth(df: pd.DataFrame, year: int, info: dict, out_dir: Path) -> Path:
    """DataFrame의 특정 연도 데이터를 DSSAT WTH 파일로 저장.

    파일명: {inst_code}{YY}01.WTH  (예: SUWO2501.WTH)
    포맷: DATE(5) + SRAD(6) + TMAX(6) + TMIN(6) + RAIN(6) = 29자 고정폭
    """
    inst_code = info["inst_code"]
    filename  = f"{inst_code}{str(year)[2:]}01.WTH"
    fpath     = out_dir / filename

    lat  = info["lat"]
    lon  = info["lon"]
    elev = info["elev"]
    # 표고 이상값 처리
    if not (0 <= elev <= _MAX_ELEV):
        elev = -99.0

    # ── 헤더 ────────────────────────────────────────────────────
    lines = []
    lines.append(
        f"*WEATHER DATA : {info['target']}, "
        f"STATION USED={info['station_en']}, S. KOREA. SOURCE: KMA\n"
    )
    lines.append("\n")
    lines.append("@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT\n")
    lines.append(
        f"{inst_code:>6s} {lat:8.3f} {lon:8.3f} "
        f"{elev:5.0f} {-99.0:5.1f} {-99.0:5.1f} {-99.0:5.1f} {-99.0:5.1f}\n"
    )
    lines.append("@DATE  SRAD  TMAX  TMIN  RAIN\n")

    # ── 데이터 행 ────────────────────────────────────────────────
    df_yr = df[df["DATE"].dt.year == year].copy()
    df_yr["WTH_DATE"] = df_yr["DATE"].dt.strftime("%y%j")
    for _, row in df_yr.iterrows():
        lines.append(
            f"{row['WTH_DATE']:>5s}"
            f"{row['SRAD']:6.1f}"
            f"{row['TMAX']:6.1f}"
            f"{row['TMIN']:6.1f}"
            f"{row['RAIN']:6.1f}\n"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    fpath.write_text("".join(lines), encoding="utf-8")
    return fpath


# ─────────────────────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────────────────────

def create_weather_station(
    location: Union[str, Tuple[float, float]],
    start_date: str,
    end_date: str,
    kma_data_key: str = "",
    kma_station_key: str = "",
    output_dir: Optional[str] = None,
    srad_factor: float = _SRAD_FACTOR,
) -> Dict:
    """KMA ASOS 데이터를 다운로드하여 DSSAT WTH 파일을 생성합니다.

    Args:
        location:      지명 문자열 (예: "Suwon") 또는 (위도, 경도) 튜플.
        start_date:    시작일 "YYYY-MM-DD".
        end_date:      종료일 "YYYY-MM-DD" (미래이면 어제로 자동 조정).
        kma_data_key:  공공데이터포털 ASOS 일별 서비스 인증키.
                       미입력 시 환경변수 KMA_DATA_API_KEY 사용.
        kma_station_key: apihub.kma.go.kr 관측소 목록 인증키.
                       미입력 시 환경변수 KMA_STATION_API_KEY 사용.
        output_dir:    WTH 파일 저장 경로. 미입력 시 DSSAT_WEATHER 환경변수
                       또는 C:/DSSAT48/Weather/ 사용.
        srad_factor:   일조시간 → 일사량 변환 계수 (기본 1.8 MJ/m²/h).

    Returns:
        {
            "ok": bool,
            "message": str,
            "wth_stem": str,        # run_simulation weather_station 인자로 사용
            "wth_files": List[str], # 생성된 파일 경로 목록
            "station": str,
            "station_id": str,
            "lat": float,
            "lon": float,
            "elev": float,
            "n_days": int,
            "date_range": str,
        }
    """
    # ── API 키 결정 ───────────────────────────────────────────────
    data_key    = kma_data_key    or os.environ.get("KMA_DATA_API_KEY", "")
    station_key = kma_station_key or os.environ.get("KMA_STATION_API_KEY", "")

    if not data_key:
        return {"ok": False, "message": "KMA_DATA_API_KEY가 없습니다. 인자 또는 환경변수로 제공하세요."}
    if not station_key:
        return {"ok": False, "message": "KMA_STATION_API_KEY가 없습니다. 인자 또는 환경변수로 제공하세요."}

    # ── 출력 디렉토리 ─────────────────────────────────────────────
    if output_dir:
        out_path = Path(output_dir)
    else:
        dssat_weather = os.environ.get("DSSAT_WEATHER", "C:/DSSAT48/Weather")
        out_path = Path(dssat_weather)

    # ── 날짜 파싱 ─────────────────────────────────────────────────
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")
    except ValueError as e:
        return {"ok": False, "message": f"날짜 형식 오류: {e}"}

    yesterday = datetime.now() - timedelta(days=1)
    if end_dt > yesterday:
        end_dt = yesterday

    if start_dt > end_dt:
        return {"ok": False, "message": f"start_date({start_date})가 end_date보다 늦습니다."}

    # ── 관측소 목록 조회 ──────────────────────────────────────────
    stations_df = _fetch_station_list(station_key)
    if stations_df is None or stations_df.empty:
        return {"ok": False, "message": "관측소 목록을 가져오지 못했습니다. KMA_STATION_API_KEY를 확인하세요."}

    # ── 위치 → 좌표 ───────────────────────────────────────────────
    if isinstance(location, tuple) and len(location) == 2:
        user_lat, user_lon = float(location[0]), float(location[1])
        target_name = f"{user_lat:.3f},{user_lon:.3f}"
    else:
        user_lat, user_lon = _geocode(str(location))
        target_name = str(location)
        if user_lat is None:
            return {"ok": False, "message": f"'{location}' 위치를 찾을 수 없습니다. geopy가 설치되었는지 확인하세요."}

    # ── 최근접 관측소 ─────────────────────────────────────────────
    station, dist_km = _nearest_station(user_lat, user_lon, stations_df)
    station_id  = str(int(float(station["지점"])))
    station_en  = station["STN_EN"]
    inst_code   = station_en.replace(" ", "").upper()[:4]
    raw_elev    = float(station["지점높이(m)"])
    elev        = raw_elev if (0 <= raw_elev <= _MAX_ELEV) else -99.0

    wth_info = {
        "inst_code":  inst_code,
        "station_en": station_en,
        "target":     target_name,
        "lat":        float(station["위도"]),
        "lon":        float(station["경도"]),
        "elev":       elev,
    }

    # ── 월별 API 루프 ─────────────────────────────────────────────
    all_items: List[dict] = []
    cur = start_dt
    while cur <= end_dt:
        month_end = cur + relativedelta(months=1, days=-1)
        req_end   = min(month_end, end_dt)
        items = _fetch_asos_monthly(
            cur.strftime("%Y%m%d"),
            req_end.strftime("%Y%m%d"),
            station_id,
            data_key,
        )
        all_items.extend(items)
        time.sleep(_RATE_LIMIT)
        cur += relativedelta(months=1)

    if not all_items:
        return {
            "ok": False,
            "message": f"관측소 {station_en}({station_id})의 해당 기간 ASOS 데이터가 없습니다.",
        }

    # ── 데이터 가공 ───────────────────────────────────────────────
    df = _build_df(all_items, srad_factor)
    if df is None or df.empty:
        return {"ok": False, "message": "데이터 처리 실패."}

    # ── 연도별 WTH 파일 생성 ──────────────────────────────────────
    created_files = []
    created_stems = []
    for year in range(start_dt.year, end_dt.year + 1):
        df_yr = df[df["DATE"].dt.year == year]
        if df_yr.empty:
            continue
        fpath = _write_wth(df, year, wth_info, out_path)
        created_files.append(str(fpath))
        created_stems.append(f"{inst_code}{str(year)[2:]}01")

    if not created_files:
        return {"ok": False, "message": "WTH 파일 생성 실패 (데이터 없음)."}

    # run_simulation에서 쓸 대표 stem (가장 최근 연도)
    primary_stem = created_stems[-1]

    actual_start = df["DATE"].min().strftime("%Y-%m-%d")
    actual_end   = df["DATE"].max().strftime("%Y-%m-%d")

    return {
        "ok": True,
        "message": f"{len(created_files)}개 WTH 파일 생성 완료 ({station_en}, {dist_km:.1f} km)",
        "wth_stem":   primary_stem,
        "wth_stems":  created_stems,
        "wth_files":  created_files,
        "station":    station_en,
        "station_id": station_id,
        "lat":        wth_info["lat"],
        "lon":        wth_info["lon"],
        "elev":       elev,
        "n_days":     len(df),
        "date_range": f"{actual_start} ~ {actual_end}",
    }
