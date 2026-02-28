# DSSAT-MCP Usage Examples

These examples show natural language prompts that can be sent to Claude (or any MCP-compatible LLM agent) after registering the DSSAT-MCP server.

---

## 1. Basic Simulation

```
Run a maize simulation at Suwon, Korea.
Sowing date: May 1, 2025.
Soil: KR_JD_MAI1.
Nitrogen: 120 kg/ha.
```

Expected output includes: grain yield (kg/ha), maturity date, growing days, harvest index, WUE.

---

## 2. Cultivar Parameter Estimation (Calibration)

### Maize
```
Estimate cultivar parameters for Korean maize (KR0003) at Suwon 2025.
Sowing date: May 1 2025.
Observed heading date: July 10 2025.
Observed maturity date: September 10 2025.
Observed yield: 7500 kg/ha.
Observed thousand-grain weight: 280 g.
Soil: KR_JD_MAI1.
```

### Winter Wheat
```
Estimate cultivar parameters for Korean wheat (KR0001) at Suwon.
Sowing date: October 25 2025.
Observed heading date: April 22 2026.
Observed maturity date: June 7 2026.
Observed yield: 5200 kg/ha.
Observed thousand-grain weight: 38 g.
Soil: KR_JD_MAI2.
Weather station: SUWO2601.
```

### Spring Barley
```
Estimate cultivar parameters for Korean barley (KR0001, Tapgol).
Sowing date: March 1 2025.
Observed heading date: May 10 2025.
Observed maturity date: June 7 2025.
Observed yield: 3500 kg/ha.
Observed thousand-grain weight: 42 g.
Soil: KR_JD_MAI2.
```

---

## 3. Model Evaluation

```
Evaluate simulation accuracy for maize yield:
- Year 1: observed 7500 kg/ha, simulated 7280 kg/ha
- Year 2: observed 6800 kg/ha, simulated 7010 kg/ha
- Year 3: observed 8100 kg/ha, simulated 7850 kg/ha

Calculate RMSE, d-index, NSE, and R².
```

---

## 4. Sensitivity Analysis

### Nitrogen response
```
Run a nitrogen sensitivity analysis for maize at Suwon 2025.
Test nitrogen rates: 0, 60, 120, 180, 240 kg/ha.
Show yield response and nitrogen use efficiency.
```

### Temperature sensitivity (climate change)
```
Run a temperature sensitivity analysis for Korean wheat at Suwon.
Sowing: October 25 2025.
Test temperature offsets: 0, +1, +2, +3, +4 °C.
Weather: SUWO2601.
```

---

## 5. Climate Change Scenarios

```
Compare RCP scenarios for maize at Suwon, sowing May 1 2025:
- Baseline (current climate)
- +2°C warming only
- RCP4.5: +2°C, CO2 = 550 ppm
- RCP8.5: +4°C, CO2 = 700 ppm
Show yield changes and ranking.
```

---

## 6. Multi-Scenario Batch (Sowing Date Optimization)

```
Find the optimal sowing date for maize at Suwon 2025.
Test sowing dates: April 20, May 1, May 10, May 20, June 1.
Nitrogen: 120 kg/ha. Soil: KR_JD_MAI1.
Rank by grain yield.
```

---

## 7. Weather Station Download (KMA)

```
Download weather data for Jeonju (전주) from January 2020 to December 2024
and create a DSSAT weather file.
KMA data API key: [your key]
KMA station API key: [your key]
```

---

## 8. List Available Resources

```
What weather stations are available near Suwon?

What Korean soil profiles are available?

What cultivars are available for barley?
```

---

## Programmatic Testing (Python)

For testing outside Claude Desktop, use the mock pattern:

```python
import types, sys

# Mock fastmcp for direct testing
fake = types.ModuleType('fastmcp')
class _FakeMCP:
    def __init__(self, *a, **kw): pass
    def tool(self, fn): return fn
    def run(self): pass
fake.FastMCP = _FakeMCP
sys.modules['fastmcp'] = fake

import dssat_mcp_server as mcp

# Run a simulation directly
result = mcp.run_simulation(
    crop="maize",
    sowing_date="2025-05-01",
    soil_profile="KR_JD_MAI1",
    nitrogen_kg_ha=120,
    weather_station="SUWO2501",
)
print(result["human_summary"])
```
