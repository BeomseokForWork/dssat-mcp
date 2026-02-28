# DSSAT-MCP: AI Agent Interface for Crop Model Calibration

A proof-of-concept MCP (Model Context Protocol) server that wraps the DSSAT CSM v4.8 crop simulation model, enabling natural language interaction through LLM agents (e.g., Claude).

> **Paper**: *"MCP-Based AI Agent Interface for Crop Model Calibration: A Proof of Concept with DSSAT"*
> (under review)

---

## Overview

Crop model calibration traditionally requires deep technical expertise: writing experiment files, configuring parameters, parsing outputs. This project demonstrates that wrapping DSSAT in an MCP server allows an LLM agent to perform calibration tasks through natural language alone.

### Available Tools (9 MCP tools)

| Tool | Description |
|---|---|
| `list_models` | List supported crops, stations, and soils |
| `list_cultivars` | List cultivars from DSSAT .CUL files |
| `list_stations` | Browse available weather files |
| `list_soils` | Browse available soil profiles |
| `run_simulation` | Run a single DSSAT simulation |
| `run_batch` | Run multiple scenarios in one batch |
| `evaluate_simulation` | Calculate RMSE, d-index, NSE, R² |
| `sensitivity_analysis` | One-at-a-time parameter sensitivity |
| `estimate_cultivar_params` | Estimate cultivar parameters from observations |
| `create_weather_station` | Download KMA weather data → WTH file |
| `get_result` | Retrieve stored simulation results |

### Supported Crops

| Crop | Model | Korean Cultivar |
|---|---|---|
| Maize | MZCER048 | KR0003 (Dacheongok) |
| Wheat | WHCER048 | KR0001, KR0002 |
| Barley | CSCER048 | KR0001 (Tapgol), KR0002 (Seodunchal) |
| Rice | RICER048 | IB0012 |
| Soybean | CRGRO048 | KR2828 |
| Potato | PTSUB048 | IB0001 |
| Sorghum | SGCER048 | IB0001 |

---

## Prerequisites

### 1. DSSAT v4.8
Download and install from [dssat.net](https://dssat.net) (free registration required).
Default install path: `C:\DSSAT48`

### 2. Python 3.10+
```bash
pip install -r requirements.txt
```

### 3. MCP-compatible client
- [Claude Desktop](https://claude.ai/download) — recommended
- Any MCP-compatible LLM client

---

## Installation

### Step 1: Clone this repository
```bash
git clone https://github.com/YOUR_USERNAME/dssat-mcp.git
cd dssat-mcp
```

### Step 2: Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Copy data files into DSSAT
```bash
# Weather files
copy data\SUWO2501.WTH  C:\DSSAT48\Weather\
copy data\SUWO2601.WTH  C:\DSSAT48\Weather\

# Soil profiles
copy data\KR.SOL        C:\DSSAT48\Soil\

# Korean cultivar parameters
copy genotype\WHCER048.CUL  C:\DSSAT48\Genotype\WHCER048.CUL
copy genotype\BACER048.CUL  C:\DSSAT48\Genotype\BACER048.CUL
copy genotype\MZCER048.CUL  C:\DSSAT48\Genotype\MZCER048.CUL
copy genotype\SBGRO048.CUL  C:\DSSAT48\Genotype\SBGRO048.CUL
```

> **Note**: The CUL files in `genotype/` contain Korean cultivar entries added to the original DSSAT files. Back up your originals before copying.

### Step 4: Configure environment variables
```bash
copy .env.example .env
# Edit .env with your paths
```

### Step 5: Register with Claude Desktop

Edit `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "dssat-mcp": {
      "command": "python",
      "args": ["C:/path/to/dssat_mcp_server.py"],
      "env": {
        "DSSAT_HOME":    "C:/DSSAT48",
        "DSSAT_BIN":     "C:/DSSAT48/DSCSM048.EXE",
        "DSSAT_WORK":    "C:/dssat_jobs",
        "DSSAT_WEATHER": "C:/DSSAT48/Weather",
        "DSSAT_SOIL":    "C:/DSSAT48/Soil"
      }
    }
  }
}
```

---

## Quick Start

Once Claude Desktop is running with the MCP server registered, you can interact naturally:

```
"Run a maize simulation for Suwon, sowing May 1 2025, 120 kg N/ha"

"Estimate cultivar parameters for Korean wheat sown Oct 25 2025,
 heading date Apr 20 2026, maturity Jun 7 2026, yield 5200 kg/ha,
 thousand grain weight 38g"

"Compare nitrogen rates 0, 60, 120, 180, 240 kg/ha for maize
 at Suwon using sensitivity analysis"

"Evaluate simulation accuracy:
 observed yield 5000, simulated 4919;
 observed yield 6200, simulated 5850"
```

---

## Data Files

### Weather (`data/`)
| File | Station | Period | Source |
|---|---|---|---|
| `SUWO2501.WTH` | Suwon, Korea (37.26°N, 126.98°E) | Jan–Dec 2025 | KMA ASOS |
| `SUWO2601.WTH` | Suwon, Korea | Oct 2025–Dec 2026 | KMA ASOS + climatology |

### Soil (`data/`)
| Profile ID | Description |
|---|---|
| `KR_JD_MAI1` | Suwon Jungdong — Silty Clay, 120 cm |
| `KR_JD_MAI2` | Suwon Jungdong — Silt Loam, 120 cm |

### Genotype (`genotype/`)
Korean cultivar parameters added to standard DSSAT .CUL files:
- **KR0003** — Maize *Dacheongok* (옥수수 다청옥)
- **KR0001, KR0002** — Wheat *Tapgol / Seodunchal* (밀 탑골/서둔찰)
- **KR0001, KR0002** — Barley *Tapgol / Seodunchal* (보리 탑골/서둔찰)
- **KR2828** — Soybean *KRUG2828* (콩)

---

## Key Features

### Calibration (`estimate_cultivar_params`)
Estimates DSSAT cultivar parameters directly from field observations — no optimization loop required:
- **P5**: GDD from heading to maturity (all crops)
- **P1V**: Vernalization days (wheat, barley)
- **G2/G3**: Kernel weight from thousand-grain weight
- Verification simulation run automatically after estimation

### Model Evaluation (`evaluate_simulation`)
Standard statistical metrics for model performance assessment:
- RMSE, MAE, MBE (bias)
- Willmott d-index
- Nash-Sutcliffe Efficiency (NSE)
- Pearson R²

### Climate Scenarios (`run_batch`)
```python
# Example: RCP scenario comparison
run_batch(crop="maize", scenarios=[
    {"label": "baseline",  "sowing_date": "2025-05-01"},
    {"label": "+2°C",      "delta_temp": 2},
    {"label": "RCP4.5",    "delta_temp": 2, "co2_ppm": 550},
    {"label": "RCP8.5",    "delta_temp": 4, "co2_ppm": 700},
])
```

---

## Limitations

- Single AI agent (Claude) — extensible to other LLM clients via MCP protocol
- Single crop model (DSSAT) — architecture supports adding APSIM, STICS, etc.
- Definition-based calibration: accurate for phenology parameters (P5, P1V), less so for yield parameters without anthesis biomass data
- Windows native; Linux/macOS require Wine

---

## System Architecture

```
User (natural language)
        │
        ▼
  LLM Agent (Claude)
        │  MCP Protocol (JSON-RPC over stdio)
        ▼
  DSSAT-MCP Server (Python / FastMCP)
        │
        ├── FileX writer (experiment file)
        ├── Weather handler (perturbation, KMA download)
        ├── Soil handler (profile lookup)
        ├── Cultivar estimator (parameter estimation)
        └── Output parser (Summary.OUT, PlantGro.OUT, ...)
              │
              ▼
         DSSAT CSM v4.8 (DSCSM048.EXE)
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{yourname2025dssat,
  title   = {MCP-Based AI Agent Interface for Crop Model Calibration:
             A Proof of Concept with DSSAT},
  author  = {Your Name et al.},
  journal = {Computers and Electronics in Agriculture},
  year    = {2025},
  note    = {under review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

DSSAT itself is subject to its own license agreement ([dssat.net](https://dssat.net)).
