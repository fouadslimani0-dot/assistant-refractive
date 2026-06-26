import streamlit as st
import pandas as pd
from fpdf import FPDF
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Assistant Réfractif – Pôle Ophtalmologie Batna",
    layout="wide",
    page_icon="👁️"
)

# ─────────────────────────────────────────
# CSS personnalisé
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .header-box {
        background: linear-gradient(135deg, #1a3a5c, #2e6da4);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
    }
    .header-box h1 { font-size: 1.8rem; margin: 0; }
    .header-box p  { font-size: 0.95rem; margin: 4px 0 0; opacity: 0.85; }

    .verdict-ok {
        background: #d4edda; border-left: 5px solid #28a745;
        padding: 12px 16px; border-radius: 8px; font-weight: bold; color: #155724;
    }
    .verdict-ko {
        background: #f8d7da; border-left: 5px solid #dc3545;
        padding: 12px 16px; border-radius: 8px; font-weight: bold; color: #721c24;
    }
    .verdict-warn {
        background: #fff3cd; border-left: 5px solid #ffc107;
        padding: 12px 16px; border-radius: 8px; font-weight: bold; color: #856404;
    }
    .card {
        background: white; border-radius: 12px;
        padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 16px;
    }
    .metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
    .metric-item {
        background: #eef2f7; border-radius: 8px;
        padding: 8px 14px; flex: 1; min-width: 120px; text-align: center;
    }
    .metric-item .val { font-size: 1.3rem; font-weight: bold; color: #1a3a5c; }
    .metric-item .lbl { font-size: 0.72rem; color: #666; text-transform: uppercase; }
    .eye-title {
        font-size: 1.2rem; font-weight: 700;
        color: #1a3a5c; border-bottom: 2px solid #2e6da4;
        padding-bottom: 6px; margin-bottom: 14px;
    }
    div[data-testid="stSidebar"] { background: #1a3a5c; }
    div[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Constantes cliniques
# ─────────────────────────────────────────
PACHYMETRIE_MIN_LASIK   = 480   # µm
PACHYMETRIE_MIN_PKR     = 450   # µm
PACHYMETRIE_MIN_ICL     = 0     # pas de seuil cornéen pour ICL
MUR_STROMAL_MIN         = 250   # µm résiduel minimum
ABLATION_REF_FACTEUR    = 15    # µm/D à zo=6.5mm (approximation Munnerlyn)
FLAP_LASIK              = 110   # µm (épaisseur volet LASIK)
RETRAIT_PKR             = 50    # µm épithélium PKR
K_MIN                   = 36.0  # D
K_MAX                   = 48.0  # D
K_FINAL_MIN             = 33.0  # D après correction
SPH_MAX_LASIK           = 10.0  # D (valeur absolue)
CYL_MAX_LASIK           = 6.0   # D (valeur absolue)
SPH_MAX_PKR             = 8.0
CYL_MAX_PKR             = 5.0
AGE_MIN                 = 21
STABILITE_ANS           = 1     # stabilité réfractive en années


# ─────────────────────────────────────────
# Moteur de décision
# ─────────────────────────────────────────
def calculer_ablation(sph, cyl, zo):
    """Profondeur d'ablation (formule Munnerlyn simplifiée)."""
    eq_sph = abs(sph) + abs(cyl) / 2
    return eq_sph * (zo / 6.5) ** 2 * ABLATION_REF_FACTEUR


def analyser_oeil(pach, sph, cyl, k, zo, age, pupille, stabilite):
    """
    Retourne un dict avec :
      - ablation, mur_lasik, mur_pkr, k_final, es
      - verdict : 'LASIK' | 'PKR' | 'ICL' | 'CONTRE-INDIQUÉ'
      - raisons : liste de chaînes
      - classe_css : 'ok' | 'ko' | 'warn'
    """
    es       = sph + cyl / 2
    ablation = calculer_ablation(sph, cyl, zo)
    k_final  = k - abs(es) * 0.8      # approximation post-ablation
    mur_lasik = pach - FLAP_LASIK - ablation
    mur_pkr   = pach - RETRAIT_PKR   - ablation

    raisons_ko    = []
    raisons_warn  = []

    # ── Contre-indications absolues ──────────────────────────────
    if age < AGE_MIN:
        raisons_ko.append(f"Âge < {AGE_MIN} ans ({age} ans)")
    if not stabilite:
        raisons_ko.append("Réfraction non stabilisée (< 1 an)")
    if k < K_MIN or k > K_MAX:
        raisons_ko.append(f"Kératométrie hors normes ({k:.1f} D → [{K_MIN}-{K_MAX}])")
    if k_final < K_FINAL_MIN:
        raisons_ko.append(f"K résiduelle trop basse ({k_final:.1f} D < {K_FINAL_MIN} D) → risque ectasie")
    if pach < 400:
        raisons_ko.append(f"Pachymétrie critique ({pach} µm)")

    # ── Évaluation LASIK ─────────────────────────────────────────
    lasik_ok = True
    if abs(sph) > SPH_MAX_LASIK:
        raisons_warn.append(f"Sphère hors zone LASIK (|{sph:.1f}| > {SPH_MAX_LASIK} D)")
        lasik_ok = False
    if abs(cyl) > CYL_MAX_LASIK:
        raisons_warn.append(f"Cylindre hors zone LASIK (|{cyl:.1f}| > {CYL_MAX_LASIK} D)")
        lasik_ok = False
    if pach < PACHYMETRIE_MIN_LASIK:
        raisons_warn.append(f"Pachymétrie insuffisante pour LASIK ({pach} µm < {PACHYMETRIE_MIN_LASIK} µm)")
        lasik_ok = False
    if mur_lasik < MUR_STROMAL_MIN:
        raisons_warn.append(f"Mur stromal LASIK insuffisant ({mur_lasik:.0f} µm < {MUR_STROMAL_MIN} µm)")
        lasik_ok = False
    if pupille and pupille > zo + 0.5:
        raisons_warn.append(f"Pupille mésopique large ({pupille:.1f} mm) vs ZO ({zo:.1f} mm) → halos")

    # ── Évaluation PKR ──────────────────────────────────────────
    pkr_ok = True
    if abs(sph) > SPH_MAX_PKR:
        pkr_ok = False
    if abs(cyl) > CYL_MAX_PKR:
        pkr_ok = False
    if pach < PACHYMETRIE_MIN_PKR:
        pkr_ok = False
    if mur_pkr < MUR_STROMAL_MIN:
        pkr_ok = False

    # ── Verdict final ────────────────────────────────────────────
    if raisons_ko:
        verdict     = "CONTRE-INDIQUÉ"
        classe_css  = "ko"
        detail      = "ICL à évaluer si pas de contre-indication pupillaire/anatomique"
    elif lasik_ok:
        verdict    = "✅ LASIK"
        classe_css = "ok"
        detail     = f"Mur stromal résiduel : {mur_lasik:.0f} µm"
    elif pkr_ok:
        verdict    = "✅ PKR"
        classe_css = "warn"
        detail     = f"Mur stromal résiduel : {mur_pkr:.0f} µm"
    else:
        verdict    = "💉 ICL recommandé"
        classe_css = "warn"
        detail     = "Chirurgie cornéenne insuffisante – envisager phaque"

    return {
        "ablation"  : ablation,
        "mur_lasik" : mur_lasik,
        "mur_pkr"   : mur_pkr,
        "k_final"   : k_final,
        "es"        : es,
        "verdict"   : verdict,
        "detail"    : detail,
        "raisons_ko"  : raisons_ko,
        "raisons_warn": raisons_warn,
        "classe_css"  : classe_css,
    }


# ─────────────────────────────────────────
# Simulation du profil d'ablation
# ─────────────────────────────────────────
def afficher_simulation(sph, cyl, zo, zt=1.25, oeil="OD"):
    x = np.linspace(-5, 5, 500)
    r = np.abs(x)
    prof_max = max(calculer_ablation(sph, cyl, zo), 0.01)

    y = np.zeros_like(x)
    zo_r = zo / 2
    mask_zo = r <= zo_r
    mask_zt = (r > zo_r) & (r <= zo_r + zt)

    y[mask_zo] = prof_max
    y[mask_zt] = prof_max * (1 - (r[mask_zt] - zo_r) / zt)

    fig, ax = plt.subplots(figsize=(6, 2.5), facecolor='#f8fafc')
    ax.set_facecolor('#f8fafc')
    ax.fill_between(x[mask_zo], y[mask_zo], color='#dc3545', alpha=0.85, label='Zone optique')
    ax.fill_between(x[mask_zt], y[mask_zt], color='#ffc107', alpha=0.75, label='Zone transition')
    ax.fill_between(x[~mask_zo & ~mask_zt], 0, color='#28a745', alpha=0.3, label='Sans ablation')
    ax.axhline(0, color='#333', linewidth=0.8)
    ax.set_title(f"Profil d'ablation – {oeil}  (profondeur max : {prof_max:.1f} µm)", fontsize=9)
    ax.set_xlabel("Rayon cornéen (mm)", fontsize=8)
    ax.set_ylabel("Ablation (µm)", fontsize=8)
    ax.tick_params(labelsize=7)
    p1 = mpatches.Patch(color='#dc3545', label='Zone optique')
    p2 = mpatches.Patch(color='#ffc107', label='Zone transition')
    p3 = mpatches.Patch(color='#28a745', alpha=0.5, label='Sans ablation')
    ax.legend(handles=[p1, p2, p3], fontsize=7, loc='upper right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────
# Génération du PDF
# ─────────────────────────────────────────
class PDF(FPDF):
    def header(self):
        self.set_fill_color(26, 58, 92)
        self.rect(0, 0, 210, 28, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font("Arial", 'B', 14)
        self.cell(0, 10, "Pôle d'Ophtalmologie Batnéen", ln=True, align='C')
        self.set_font("Arial", '', 9)
        self.cell(0, 8, "Assistant Décisionnel en Chirurgie Réfractive", ln=True, align='C')
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", 'I', 7)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Document généré le {date.today().strftime('%d/%m/%Y')} – Usage médical interne uniquement", align='C')


def creer_pdf(nom, age, dob, donnees_yeux):
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Infos patient ────────────────────────────────────────────
    pdf.set_font("Arial", 'B', 11)
    pdf.set_fill_color(230, 238, 248)
    pdf.cell(0, 8, "  Informations Patient", fill=True, ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(60, 7, f"Nom : {nom}")
    pdf.cell(60, 7, f"Âge : {age} ans")
    pdf.cell(0,  7, f"Date de naissance : {dob if dob else '—'}", ln=True)
    pdf.cell(0,  7, f"Date d'examen : {date.today().strftime('%d/%m/%Y')}", ln=True)
    pdf.ln(4)

    for eye, d in donnees_yeux.items():
        # ── Titre œil ───────────────────────────────────────────
        pdf.set_font("Arial", 'B', 11)
        pdf.set_fill_color(46, 109, 164)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 8, f"  Œil {eye}", fill=True, ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        # ── Tableau paramètres ───────────────────────────────────
        pdf.set_font("Arial", 'B', 9)
        pdf.set_fill_color(210, 225, 242)
        headers = ["Paramètre", "Valeur", "Résultat calculé", "Valeur"]
        widths  = [50, 30, 60, 45]
        for h, w in zip(headers, widths):
            pdf.cell(w, 7, h, border=1, fill=True, align='C')
        pdf.ln()

        pdf.set_font("Arial", '', 9)
        rows = [
            ("Pachymétrie", f"{d['pach']} µm",  "Equivalent sphérique", f"{d['es']:+.2f} D"),
            ("Sphère",       f"{d['sph']:+.2f} D", "Ablation estimée",   f"{d['ablation']:.1f} µm"),
            ("Cylindre",     f"{d['cyl']:+.2f} D", "Mur stromal LASIK",  f"{d['mur_lasik']:.0f} µm"),
            ("K moyen",      f"{d['k']:.2f} D",    "Mur stromal PKR",    f"{d['mur_pkr']:.0f} µm"),
            ("Zone optique", f"{d['zo']:.1f} mm",  "K finale estimée",   f"{d['k_final']:.2f} D"),
        ]
        fill = False
        for r in rows:
            pdf.set_fill_color(245, 249, 254) if fill else pdf.set_fill_color(255, 255, 255)
            for val, w in zip(r, widths):
                pdf.cell(w, 6, str(val), border=1, fill=True)
            pdf.ln()
            fill = not fill

        pdf.ln(3)

        # ── Verdict ─────────────────────────────────────────────
        verdict = d['verdict']
        if "✅" in verdict:
            pdf.set_fill_color(212, 237, 218); pdf.set_text_color(21, 87, 36)
        elif "💉" in verdict:
            pdf.set_fill_color(255, 243, 205); pdf.set_text_color(133, 100, 4)
        else:
            pdf.set_fill_color(248, 215, 218); pdf.set_text_color(114, 28, 36)

        pdf.set_font("Arial", 'B', 10)
        # Remove emoji for PDF (latin-1 encoding limitation)
        verdict_clean = verdict.replace("✅", "").replace("💉", "").replace("❌", "").strip()
        pdf.cell(0, 8, f"Verdict : {verdict_clean}  –  {d['detail']}", fill=True, ln=True, border=1)
        pdf.set_text_color(0, 0, 0)

        # ── Alertes ─────────────────────────────────────────────
        if d['raisons_ko'] or d['raisons_warn']:
            pdf.set_font("Arial", 'B', 9)
            pdf.cell(0, 6, "Alertes cliniques :", ln=True)
            pdf.set_font("Arial", '', 9)
            for r in d['raisons_ko']:
                pdf.set_text_color(180, 0, 0)
                pdf.cell(0, 5, f"  [!] {r}", ln=True)
            for r in d['raisons_warn']:
                pdf.set_text_color(150, 100, 0)
                pdf.cell(0, 5, f"  [~] {r}", ln=True)
            pdf.set_text_color(0, 0, 0)

        pdf.ln(6)

    # ── Signature ────────────────────────────────────────────────
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Ce document est un outil d'aide à la décision. Le choix thérapeutique reste de la responsabilité du chirurgien.", ln=True, align='C')

    return pdf.output(dest='S').encode('latin-1')


# ─────────────────────────────────────────
# SIDEBAR – Paramètres globaux
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Paramètres généraux")
    st.markdown("---")
    nom     = st.text_input("Nom du patient", placeholder="Nom Prénom")
    age     = st.number_input("Âge (ans)", min_value=0, max_value=100, value=30)
    dob     = st.text_input("Date de naissance (JJ/MM/AAAA)", placeholder="01/01/1994")
    st.markdown("---")
    stabilite = st.checkbox("Réfraction stable depuis ≥ 1 an", value=True)
    st.markdown("---")
    st.markdown("### ⚙️ Paramètres chirurgicaux")
    zt_global = st.slider("Zone de transition (mm)", 0.5, 2.0, 1.25, 0.25)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; opacity:0.7; line-height:1.5'>
    <b>Seuils utilisés</b><br>
    Mur stromal min. : 250 µm<br>
    Volet LASIK : 110 µm<br>
    Épithélium PKR : 50 µm<br>
    K min/max : 36–48 D<br>
    K finale min. : 33 D
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# EN-TÊTE
# ─────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <h1>👁️ Assistant Décisionnel en Chirurgie Réfractive</h1>
  <p>Pôle d'Ophtalmologie – CHU Batna &nbsp;|&nbsp; Outil d'aide à la décision clinique</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SAISIE & RÉSULTATS PAR ŒIL
# ─────────────────────────────────────────
col_od, col_og = st.columns(2, gap="large")
resultats_pdf = {}

for eye, col in [("OD", col_od), ("OG", col_og)]:
    with col:
        st.markdown(f'<div class="eye-title">{"🔵 Œil Droit (OD)" if eye=="OD" else "🟢 Œil Gauche (OG)"}</div>', unsafe_allow_html=True)

        with st.container():
            pach    = st.number_input(f"Pachymétrie (µm)",      300, 700, 510, key=f"pach_{eye}")
            sph     = st.number_input(f"Sphère (D)",             -20.0, 10.0, -3.0, 0.25, key=f"sph_{eye}")
            cyl     = st.number_input(f"Cylindre (D)",           -8.0,  4.0, -1.0, 0.25, key=f"cyl_{eye}")
            k       = st.number_input(f"K moyen (D)",             30.0,  60.0, 43.5, 0.1, key=f"k_{eye}")
            pupille = st.number_input(f"Pupille mésopique (mm)", 3.0, 9.0, 5.5, 0.1, key=f"pup_{eye}")
            zo      = st.slider(f"Zone optique (mm)",             6.0, 7.0, 6.5, 0.1, key=f"zo_{eye}")

        # ── Calcul ─────────────────────────────────────────────
        res = analyser_oeil(pach, sph, cyl, k, zo, age, pupille, stabilite)

        # ── Métriques ──────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-item"><div class="val">{res['es']:+.2f} D</div><div class="lbl">Eq. sphérique</div></div>
          <div class="metric-item"><div class="val">{res['ablation']:.0f} µm</div><div class="lbl">Ablation</div></div>
          <div class="metric-item"><div class="val">{res['mur_lasik']:.0f} µm</div><div class="lbl">Mur LASIK</div></div>
          <div class="metric-item"><div class="val">{res['mur_pkr']:.0f} µm</div><div class="lbl">Mur PKR</div></div>
          <div class="metric-item"><div class="val">{res['k_final']:.1f} D</div><div class="lbl">K finale</div></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Verdict ────────────────────────────────────────────
        css = res['classe_css']
        st.markdown(f"""
        <div class="verdict-{css}">
            {res['verdict']}<br>
            <span style="font-weight:normal;font-size:0.85rem">{res['detail']}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Alertes ────────────────────────────────────────────
        if res['raisons_ko']:
            with st.expander("⛔ Contre-indications", expanded=True):
                for r in res['raisons_ko']:
                    st.error(r)
        if res['raisons_warn']:
            with st.expander("⚠️ Points de vigilance"):
                for r in res['raisons_warn']:
                    st.warning(r)

        # ── Simulation ─────────────────────────────────────────
        with st.expander("📊 Profil d'ablation", expanded=False):
            afficher_simulation(sph, cyl, zo, zt_global, eye)

        # Stockage pour PDF
        resultats_pdf[eye] = {
            "pach": pach, "sph": sph, "cyl": cyl, "k": k, "zo": zo,
            **res
        }

# ─────────────────────────────────────────
# RÉSUMÉ COMPARATIF
# ─────────────────────────────────────────
st.divider()
st.markdown("### 📋 Résumé comparatif")

rows_cmp = {
    "Équivalent sphérique": [f"{resultats_pdf[e]['es']:+.2f} D" for e in ["OD","OG"]],
    "Ablation estimée":     [f"{resultats_pdf[e]['ablation']:.0f} µm" for e in ["OD","OG"]],
    "Mur stromal LASIK":    [f"{resultats_pdf[e]['mur_lasik']:.0f} µm" for e in ["OD","OG"]],
    "Mur stromal PKR":      [f"{resultats_pdf[e]['mur_pkr']:.0f} µm" for e in ["OD","OG"]],
    "K finale estimée":     [f"{resultats_pdf[e]['k_final']:.2f} D" for e in ["OD","OG"]],
    "Verdict":              [resultats_pdf[e]['verdict'] for e in ["OD","OG"]],
}
df_cmp = pd.DataFrame(rows_cmp, index=["OD", "OG"]).T
st.dataframe(df_cmp, use_container_width=True)

# ─────────────────────────────────────────
# GÉNÉRATION PDF
# ─────────────────────────────────────────
st.divider()
col_btn1, col_btn2 = st.columns([1, 3])
with col_btn1:
    if st.button("📄 Générer le rapport PDF", type="primary", use_container_width=True):
        if not nom.strip():
            st.error("Veuillez saisir le nom du patient avant de générer le rapport.")
        else:
            with st.spinner("Génération du rapport…"):
                pdf_bytes = creer_pdf(nom, age, dob, resultats_pdf)
            st.download_button(
                label="⬇️ Télécharger le rapport PDF",
                data=pdf_bytes,
                file_name=f"Rapport_Refractif_{nom.replace(' ','_')}_{date.today().isoformat()}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
with col_btn2:
    st.caption("Le rapport PDF inclut toutes les données biométriques, les calculs, le verdict et les alertes cliniques.")
