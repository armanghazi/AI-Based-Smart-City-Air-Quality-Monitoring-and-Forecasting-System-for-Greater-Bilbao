"""
dashboard/pdf_report.py

PDF report generator — shared utility used by:
  - pages/0_Daily_Briefing.py              -> Daily Alert Report (1 page)
  - pages/6_Smart_City_Decision_Support.py -> Monthly Risk Report (3 pages)

Dependencies: fpdf2 (pip install fpdf2)

NOTE: All text is passed through _safe() before rendering to strip Unicode
characters unsupported by Helvetica (em-dash, mu, subscripts, emoji, etc.).
"""

from __future__ import annotations

from datetime import date
from io import BytesIO

import pandas as pd
from fpdf import FPDF

# --------------------------------------------------
# BRAND COLOURS (RGB tuples)
# --------------------------------------------------

C_DARK   = (26,  26,  46)
C_WHITE  = (255, 255, 255)
C_GREEN  = (46,  204, 113)
C_YELLOW = (243, 156,  18)
C_RED    = (231,  76,  60)
C_BLUE   = (41,  128, 185)
C_LIGHT  = (245, 246, 248)
C_GREY   = (150, 150, 150)


# --------------------------------------------------
# UNICODE SANITISER
# Must be called on EVERY string before fpdf cell/multi_cell.
# --------------------------------------------------

def _safe(text) -> str:
    """Replace all characters outside Helvetica Latin-1 range."""
    return (
        str(text)
        .replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u00d7", "x")
        .replace("\u00b5", "u")
        .replace("\u00b2", "2")
        .replace("\u00b3", "3")
        .replace("\u2082", "2")
        .replace("\u2083", "3")
        .replace("\u00b0", "deg")
        .replace("\u00b7", ".")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2022", "-")
        .replace("\u2026", "...")
        .replace("\u03bc", "u")
        .replace("x", "x")
        .replace("*", "*")
        # common substitutions
        .replace("x", "x")
        .replace("u", "u")
        .replace("--", "-")
        # emoji / symbols
        .replace("\u2714", "OK")
        .replace("\u2705", "OK")
        .replace("\u26a0\ufe0f", "!")
        .replace("\u26a0", "!")
        .replace("\U0001f7e2", "")
        .replace("\U0001f7e1", "")
        .replace("\U0001f534", "")
        .replace("\u2713", "OK")
        # pollutant subscripts
        .replace("NO\u2082", "NO2")
        .replace("PM\u2082.\u2085", "PM2.5")
        .replace("SO\u2082", "SO2")
        # zone icons
        .replace("\U0001f3d9\ufe0f", "")
        .replace("\U0001f3ed", "")
        .replace("\u2693", "")
        .replace("\U0001f30a", "")
        .replace("\U0001f6e2\ufe0f", "")
        # remaining unicode symbols
        .replace("\u00d7", "x")
        .replace("\u00b7", ".")
        .replace("\u00b5", "u")
        # ASCII fallbacks for common chars
        .replace(chr(215), "x")
        .replace(chr(183), ".")
        .replace(chr(181), "u")
        .replace(chr(8212), "-")
        .replace(chr(8211), "-")
        .replace(chr(215), "x")
    )


# --------------------------------------------------
# COLOUR HELPERS
# --------------------------------------------------

def _risk_color(score: float) -> tuple:
    if score < 100:   return C_GREEN
    elif score < 200: return C_YELLOW
    return C_RED


def _ratio_color(ratio) -> tuple:
    if ratio is None: return C_BLUE
    try:
        r = float(ratio)
    except (ValueError, TypeError):
        return C_BLUE
    if r > 2:   return C_RED
    if r > 1:   return C_YELLOW
    return C_GREEN


# --------------------------------------------------
# BASE PDF CLASS
# --------------------------------------------------

class AirQualityPDF(FPDF):

    def __init__(self, title: str, subtitle: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        # Sanitise BEFORE storing — header() is called inside add_page()
        self.report_title    = _safe(title)
        self.report_subtitle = _safe(subtitle)
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)
        self.add_page()

    def header(self):
        self.set_fill_color(*C_DARK)
        self.rect(0, 0, 210, 22, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*C_WHITE)
        self.set_xy(15, 5)
        self.cell(0, 8, self.report_title, ln=False)
        self.set_font("Helvetica", "", 8)
        self.set_xy(15, 13)
        self.cell(0, 5, self.report_subtitle, ln=False)
        self.set_text_color(*C_DARK)
        self.ln(18)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*C_GREY)
        self.cell(
            0, 5,
            _safe(
                "GeoAI Smart City Platform - Greater Bilbao - "
                "Data: Basque Government (CC BY 4.0) + Open-Meteo (CC BY 4.0) - "
                f"geoai-dashboard.streamlit.app   |   Page {self.page_no()}"
            ),
            align="C",
        )

    def section_title(self, text: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*C_DARK)
        self.set_fill_color(232, 234, 237)
        self.cell(0, 8, _safe(f"  {text}"), ln=True, fill=True)
        self.ln(2)

    def body_text(self, text: str, size: int = 9):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*C_DARK)
        self.multi_cell(0, 5, _safe(text))
        self.ln(2)

    def key_value(self, key: str, value: str, color=None):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*C_DARK)
        self.cell(60, 6, _safe(key), ln=False)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*(color if color else C_DARK))
        self.cell(0, 6, _safe(value), ln=True)
        self.set_text_color(*C_DARK)

    def table_header(self, cols: list):
        self.set_fill_color(*C_DARK)
        self.set_text_color(*C_WHITE)
        self.set_font("Helvetica", "B", 8)
        for label, w in cols:
            self.cell(w, 7, _safe(label), border=0, fill=True, align="C")
        self.ln()
        self.set_text_color(*C_DARK)

    def table_row(self, values: list, widths: list, fill=False, colors=None):
        self.set_fill_color(*C_LIGHT)
        self.set_font("Helvetica", "", 8)
        for i, (val, w) in enumerate(zip(values, widths)):
            c = colors[i] if (colors and colors[i]) else C_DARK
            self.set_text_color(*c)
            self.cell(w, 6, _safe(val), border="B", fill=fill, align="C")
        self.ln()
        self.set_text_color(*C_DARK)

    def disclaimer(self, text: str):
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*C_GREY)
        self.multi_cell(0, 4, _safe(text))
        self.set_text_color(*C_DARK)


# ==================================================
# REPORT 1 — DAILY ALERT REPORT
# ==================================================

def generate_daily_report(
    latest_date:    pd.Timestamp,
    current_values: dict,
    fc_df:          pd.DataFrame,
    zone_action:    str,
    worst_zone:     str,
    who_annual:     dict,
    eu_annual:      dict,
    alert_limits:   dict,
) -> bytes:
    tomorrow = latest_date + pd.Timedelta(days=1)

    pdf = AirQualityPDF(
        title="Daily Air Quality Alert Report - Greater Bilbao",
        subtitle=(
            f"Forecast date: {tomorrow.strftime('%d %B %Y')}  |  "
            f"Based on data: {latest_date.strftime('%d %B %Y')}  |  "
            "Alert standard: EU Directive 2008/50/EC"
        ),
    )

    # Section 1 — Today averages
    pdf.section_title("1. Today City-Wide Averages")
    cols1 = [
        ("Pollutant", 32), ("Today ug/m3", 32),
        ("WHO limit", 30), ("EU limit", 30),
        ("vs WHO", 28), ("vs EU", 28),
    ]
    pdf.table_header(cols1)
    w1 = [c[1] for c in cols1]

    for i, p in enumerate(["PM2.5", "PM10", "NO2", "SO2"]):
        val     = current_values.get(p, float("nan"))
        who_lim = who_annual.get(p, 40.0)
        eu_lim  = eu_annual.get(p)
        who_r   = val / who_lim if who_lim and not pd.isna(val) else None
        eu_r    = val / eu_lim  if eu_lim  and not pd.isna(val) else None
        pdf.table_row(
            [
                p,
                f"{val:.1f}" if not pd.isna(val) else "N/A",
                f"{who_lim:.0f}",
                f"{eu_lim:.0f}" if eu_lim else "-",
                f"{who_r:.1f}x" if who_r else "-",
                f"{eu_r:.1f}x"  if eu_r  else "-",
            ],
            w1,
            fill=(i % 2 == 0),
            colors=[None, None, None, None,
                    _ratio_color(who_r), _ratio_color(eu_r)],
        )

    pdf.ln(4)

    # Section 2 — Forecast exceedances
    pdf.section_title("2. Tomorrow Forecast - EU Directive Exceedances")
    exceed_df = fc_df[fc_df["Exceeds"]] if not fc_df.empty else pd.DataFrame()

    if exceed_df.empty:
        pdf.body_text(
            f"No EU Directive exceedances forecast for "
            f"{tomorrow.strftime('%d %B %Y')}. "
            "All stations within legally binding limits."
        )
    else:
        pdf.body_text(
            f"WARNING: {len(exceed_df)} EU Directive exceedance(s) "
            f"forecast for {tomorrow.strftime('%d %B %Y')}."
        )
        cols2 = [
            ("Station", 40), ("Zone", 30), ("Pollutant", 22),
            ("Forecast ug/m3", 34), ("EU Limit", 26), ("Ratio", 28),
        ]
        pdf.table_header(cols2)
        w2 = [c[1] for c in cols2]
        for i, (_, row) in enumerate(exceed_df.iterrows()):
            ratio = row.get("Ratio")
            lim   = row.get("Limit")
            stn   = str(row.get("station", row.get("Station", "-"))).split("_")[0]
            pdf.table_row(
                [
                    stn,
                    str(row.get("Zone", "-")),
                    str(row.get("Pollutant", "-")),
                    f"{row.get('Forecast', 0):.1f}",
                    f"{lim:.0f}" if pd.notna(lim) else "-",
                    f"{ratio:.2f}x" if pd.notna(ratio) else "-",
                ],
                w2,
                fill=(i % 2 == 0),
                colors=[None, None, None, None, None, _ratio_color(ratio)],
            )

    pdf.ln(4)

    # Section 3 — Recommended action
    pdf.section_title(f"3. Recommended Action - {_safe(worst_zone)} Zone")
    pdf.body_text(zone_action)
    pdf.ln(2)
    pdf.disclaimer(
        "Note: Forecasts use XGBoost models (test R2=0.39-0.56). "
        "Alert thresholds use EU Directive 2008/50/EC limits. "
        "For WHO-based health analysis see the Urban Risk Index page. "
        "This report is for indicative purposes only."
    )

    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# ==================================================
# REPORT 2 — MONTHLY RISK REPORT
# ==================================================

def generate_monthly_report(
    latest_date:       pd.Timestamp,
    window_label:      str,
    station_means:     pd.DataFrame,
    alerts_df:         pd.DataFrame,
    zone_means:        pd.DataFrame,
    summary_text:      str,
    n_above:           int,
    worst_station:     pd.Series,
    worst_pollutant:   str,
    city_ratios:       dict,
    who_annual:        dict,
    eu_annual:         dict,
    zone_meta:         dict,
    zone_action_tiers: dict,
) -> bytes:
    forecast_date = latest_date + pd.Timedelta(days=1)

    pdf = AirQualityPDF(
        title="Air Quality Risk Report - Greater Bilbao",
        subtitle=(
            f"Assessment window: {window_label}  |  "
            f"Generated: {date.today().strftime('%d %B %Y')}  |  "
            "Standards: WHO 2021 + EU Directive 2008/50/EC"
        ),
    )

    # Section 1 — Executive Summary
    pdf.section_title("1. Executive Summary")
    pdf.body_text(summary_text.strip())
    pdf.ln(2)

    kpis = [
        ("Stations above WHO:", f"{n_above} / {len(station_means)}"),
        ("Highest-risk station:",
         f"{worst_station['station']} (score {worst_station['RiskScore']:.0f})"),
        ("Most critical pollutant:",
         f"{worst_pollutant} - {city_ratios[worst_pollutant]:.1f}x WHO"),
        ("Assessment window:", window_label),
        ("Latest data date:", latest_date.strftime("%d %B %Y")),
        ("Forecast date:", forecast_date.strftime("%d %B %Y")),
    ]
    for key, val in kpis:
        pdf.key_value(key, val)
    pdf.ln(4)

    # Section 2 — Station ranking
    pdf.section_title("2. Station Risk Ranking (WHO 2021 composite score)")
    pdf.body_text(
        "Composite risk = mean(concentration / WHO limit) x 100 "
        "across PM2.5, PM10, NO2. Score > 100 = above WHO guideline."
    )
    cols3 = [
        ("Rank", 12), ("Station", 36), ("Zone", 26),
        ("PM2.5", 20), ("PM10", 20), ("NO2", 20),
        ("Score", 22), ("Level", 24),
    ]
    pdf.table_header(cols3)
    w3 = [c[1] for c in cols3]
    for i, (_, row) in enumerate(station_means.iterrows()):
        score = row["RiskScore"]
        level = str(row["RiskLevel"]).replace(" WHO guideline", "")
        pdf.table_row(
            [
                str(i + 1),
                str(row["station"]).split("_")[0],
                str(row["Zone"]),
                f"{row['PM2.5']:.1f}",
                f"{row['PM10']:.1f}",
                f"{row['NO2']:.1f}",
                f"{score:.0f}",
                level,
            ],
            w3,
            fill=(i % 2 == 0),
            colors=[None, None, None, None, None, None,
                    _risk_color(score), _risk_color(score)],
        )
    pdf.ln(4)

    # Section 3 — WHO vs EU
    pdf.section_title("3. WHO 2021 vs EU Directive 2008/50/EC")
    pdf.body_text(
        "WHO 2021 guidelines are 2-5x stricter than legally binding EU limits. "
        "Exceeding WHO does NOT constitute a legal violation in Spain."
    )
    cols4 = [
        ("Pollutant", 22), ("City Mean ug/m3", 30),
        ("WHO limit", 24), ("vs WHO", 20),
        ("EU limit", 24), ("vs EU", 20), ("EU Status", 40),
    ]
    pdf.table_header(cols4)
    w4 = [c[1] for c in cols4]
    for i, p in enumerate(["PM2.5", "PM10", "NO2", "SO2"]):
        cm     = float(station_means[p].mean()) if p in station_means.columns else float("nan")
        who_l  = who_annual.get(p, 40.0)
        eu_l   = eu_annual.get(p, 125.0)
        who_r  = cm / who_l if who_l and not pd.isna(cm) else None
        eu_r   = cm / eu_l  if eu_l  and not pd.isna(cm) else None
        eu_st  = "Above EU" if (eu_r and eu_r > 1) else "Within EU"
        pdf.table_row(
            [
                p,
                f"{cm:.1f}" if not pd.isna(cm) else "N/A",
                f"{who_l:.0f}",
                f"{who_r:.1f}x" if who_r else "-",
                f"{eu_l:.0f}",
                f"{eu_r:.1f}x"  if eu_r  else "-",
                eu_st,
            ],
            w4,
            fill=(i % 2 == 0),
            colors=[None, None, None, _ratio_color(who_r),
                    None, _ratio_color(eu_r),
                    C_RED if (eu_r and eu_r > 1) else C_GREEN],
        )
    pdf.ln(4)

    # Page 2 — Forecast alerts
    pdf.add_page()
    pdf.section_title(
        f"4. Tomorrow Forecast Alerts - {forecast_date.strftime('%d %B %Y')}"
    )
    if alerts_df.empty:
        pdf.body_text("No forecast data available.")
    else:
        exceed = alerts_df[alerts_df["Status"].str.contains("Above")]
        if exceed.empty:
            pdf.body_text(
                "No WHO exceedances forecast. All stations within WHO guidelines."
            )
        else:
            pdf.body_text(
                f"{len(exceed)} WHO exceedance(s) forecast. "
                "Review affected stations."
            )
        cols5 = [
            ("Station", 34), ("Zone", 24), ("Pollutant", 20),
            ("Forecast ug/m3", 30), ("WHO limit", 24),
            ("Ratio", 22), ("Status", 26),
        ]
        pdf.table_header(cols5)
        w5 = [c[1] for c in cols5]
        for i, (_, row) in enumerate(
            alerts_df.sort_values("Ratio", ascending=False).iterrows()
        ):
            ratio  = row.get("Ratio")
            lim    = row.get("WHO limit")
            status = "ABOVE" if str(row["Status"]).startswith("⚠") else "OK"
            pdf.table_row(
                [
                    str(row["Station"]).split("_")[0],
                    str(row["Zone"]),
                    str(row["Pollutant"]),
                    f"{row['Forecast']:.1f}",
                    f"{lim:.0f}" if pd.notna(lim) else "-",
                    f"{ratio:.2f}x" if pd.notna(ratio) else "-",
                    status,
                ],
                w5,
                fill=(i % 2 == 0),
                colors=[None, None, None, None, None,
                        _ratio_color(ratio),
                        C_RED if status == "ABOVE" else C_GREEN],
            )
    pdf.ln(4)

    # Section 5 — Zone recommendations
    pdf.section_title("5. Zone-Level Recommended Actions")

    def _tier(ratio) -> str:
        try:
            r = float(ratio) if ratio is not None else 0
        except (ValueError, TypeError):
            return "low"
        if r > 2: return "high"
        if r > 1: return "mid"
        return "low"

    tier_label = {"low": "Routine", "mid": "Elevated", "high": "Action required"}
    tier_color = {"low": C_GREEN,   "mid": C_YELLOW,   "high": C_RED}

    for zone, meta in zone_meta.items():
        if zone_means is None or zone_means.empty or zone not in zone_means.index:
            continue
        zrow    = zone_means.loc[zone]
        key_p   = meta["key_pollutant"]
        who_l   = who_annual.get(key_p, 40.0)
        val     = float(zrow.get(key_p, float("nan")))
        ratio   = val / who_l if who_l and not pd.isna(val) else None
        t       = _tier(ratio)
        action  = zone_action_tiers.get(zone, {}).get(t, "-")

        ratio_s = f"({ratio:.1f}x WHO)" if ratio else ""
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*C_DARK)
        pdf.cell(0, 6, _safe(f"{zone} - {key_p}: {val:.1f} ug/m3 {ratio_s}"), ln=True)

        pdf.set_fill_color(*tier_color[t])
        pdf.set_text_color(*C_WHITE)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(34, 5, tier_label[t], fill=True, align="C")
        pdf.set_text_color(*C_DARK)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(0, 5, _safe(f"  {action}"), ln=True)
        pdf.ln(2)

    pdf.ln(3)
    pdf.disclaimer(
        "Disclaimer: Forecasts use XGBoost models trained on 2015-2022 data "
        "(test R2=0.39-0.56). This report is for indicative purposes only and "
        "does not constitute official legal compliance documentation. "
        "For regulatory compliance refer to the official Basque Government "
        "Red de Control de Calidad del Aire (RVCA) reports."
    )

    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()