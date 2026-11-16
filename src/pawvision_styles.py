import streamlit as st


def apply_styles(dark: bool) -> None:
    colors = {
        "bg": "#11151e" if dark else "#f8f9f7",
        "surface": "#1b202c" if dark else "#ffffff",
        "sidebar": "#171c26" if dark else "#ffffff",
        "text": "#eff1f7" if dark else "#222b39",
        "muted": "#a9b1c1" if dark else "#626d7b",
        "accent": "#b9aae9" if dark else "#6b58ac",
        "accent_soft": "#332d49" if dark else "#f0edfa",
        "border": "#303847" if dark else "#e8ebe9",
        "soft": "#272d3b" if dark else "#f4f5f2",
        "purple_soft": "#332d49" if dark else "#f0edfa",
        "teal_soft": "#203b3b" if dark else "#e8f4f0",
    }
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {{ --pv-bg:{colors['bg']}; --pv-surface:{colors['surface']}; --pv-sidebar:{colors['sidebar']}; --pv-text:{colors['text']}; --pv-muted:{colors['muted']}; --pv-accent:{colors['accent']}; --pv-accent-soft:{colors['accent_soft']}; --pv-border:{colors['border']}; --pv-soft:{colors['soft']}; --pv-purple-soft:{colors['purple_soft']}; --pv-teal-soft:{colors['teal_soft']}; }}
    html, body, [class*="css"], .stApp {{ font-family: 'DM Sans', sans-serif; }}
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background: var(--pv-bg); color: var(--pv-text); }}
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{ background: var(--pv-sidebar); border-right: 1px solid var(--pv-border); }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: .8rem; }}
    [data-testid="stSidebar"] .stRadio > div {{ gap: .18rem; }}
    [data-testid="stSidebar"] .stRadio label {{ padding: .68rem .8rem; border-radius: 9px; color: var(--pv-muted); font-weight: 600; }}
    [data-testid="stSidebar"] .stRadio label:has(input:checked) {{ background: var(--pv-purple-soft); color: var(--pv-accent); }}
    [data-testid="stSidebar"] .stRadio label:hover {{ background: var(--pv-soft); }}
    [data-testid="stSidebar"] .stRadio label > div:first-child {{ display: none; }}
    [data-testid="stMainBlockContainer"] {{ max-width: 1390px; padding: 2.8rem 3.2rem 4rem; }}
    h1, h2, h3, h4 {{ font-family: 'Manrope', sans-serif !important; color: var(--pv-text) !important; letter-spacing: -.035em; }}
    p, label, [data-testid="stMarkdownContainer"] {{ color: var(--pv-text); }}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: var(--pv-muted) !important; }}
    .brand {{ display:flex; align-items:center; gap:11px; padding: 8px 0 22px; }}
    .brand-tagline {{ margin:-16px 0 20px 49px; color:var(--pv-muted); font-size:10px; letter-spacing:.015em; }}
    .brand-mark {{ display:grid; place-items:center; width:38px; height:38px; border-radius:12px; background:#8372ca; color:white; font-size:20px; font-weight:800; }}
    .brand-name {{ font-family:Manrope,sans-serif; font-size:18px; font-weight:800; letter-spacing:-.05em; color:var(--pv-text); }}
    .brand-name span {{ color:#8a7acb; }}
    .sidebar-label, .eyebrow {{ color:var(--pv-accent); font-size:10px; letter-spacing:.16em; font-weight:800; text-transform:uppercase; }}
    .sidebar-label {{ padding: 12px 0 0 9px; color:var(--pv-muted); }}
    .sidebar-note {{ margin-top:20px; padding:16px; border:1px solid var(--pv-border); border-radius:12px; background:var(--pv-soft); color:var(--pv-muted); font-size:12px; line-height:1.6; }}
    .sidebar-note strong {{ display:block; color:var(--pv-text); font-size:12px; margin-bottom:3px; }}
    .topline {{ display:flex; align-items:center; justify-content:space-between; padding-bottom:20px; border-bottom:1px solid var(--pv-border); color:var(--pv-muted); font-size:12px; font-weight:600; }}
    .topline strong {{ color:var(--pv-text); }}
    .status-dot {{ display:inline-block; width:7px; height:7px; background:#60b49e; border-radius:50%; margin-right:7px; }}
    .hero-wrap {{ background:var(--pv-surface); border:1px solid var(--pv-border); border-radius:18px; padding:36px 38px; min-height:245px; position:relative; overflow:hidden; }}
    .hero-wrap h1 {{ font-size: clamp(30px, 3.4vw, 45px); line-height:1.14; max-width:540px; margin:15px 0 12px; }}
    .hero-wrap p {{ color:var(--pv-muted); font-size:14px; max-width:490px; line-height:1.7; margin:0; }}
    .hero-art {{ position:absolute; right:4%; top:26px; width:235px; height:205px; border-radius:50%; background:var(--pv-purple-soft); }}
    .hero-art:before, .hero-art:after {{ content:''; position:absolute; border:1px solid #a99ad4; border-radius:50%; opacity:.3; }}
    .hero-art:before {{ inset:20px; }} .hero-art:after {{ inset:40px; }}
    .hero-art svg {{ position:absolute; z-index:1; left:38px; top:27px; width:165px; height:150px; }}
    .section-header {{ display:flex; align-items:end; justify-content:space-between; margin:26px 0 14px; }}
    .section-header h2 {{ font-size:19px; margin:0; }} .section-header span {{ font-size:12px; color:var(--pv-muted); }}
    .feature-card, .metric-card, .panel, .step-card {{ background:var(--pv-surface); border:1px solid var(--pv-border); border-radius:14px; }}
    .feature-card {{ min-height:170px; padding:20px; transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease; }}
    .feature-card:hover, .metric-card:hover {{ transform:translateY(-2px); border-color:#c9c0e2; box-shadow:0 8px 24px rgba(33,38,52,.06); }}
    .feature-icon {{ display:grid; place-items:center; width:37px; height:37px; border-radius:11px; background:var(--pv-purple-soft); color:var(--pv-accent); font-size:19px; margin-bottom:16px; }}
    .feature-card strong {{ font-family:Manrope,sans-serif; font-size:13px; color:var(--pv-text); }}
    .feature-card p {{ font-size:12px; color:var(--pv-muted); line-height:1.6; margin:7px 0 0; }}
    .metric-card {{ padding:18px; min-height:105px; }} .metric-card small {{ color:var(--pv-muted); font-size:11px; font-weight:600; }}
    .metric-card strong {{ display:block; color:var(--pv-text); font-family:Manrope,sans-serif; font-size:24px; margin-top:10px; letter-spacing:-.04em; }}
    .metric-card strong em {{ color:#65ad9a; font-size:11px; font-style:normal; margin-left:5px; }}
    .page-intro {{ padding: 28px 0 16px; }} .page-intro h1 {{ font-size:34px; margin:8px 0; }} .page-intro p {{ color:var(--pv-muted); font-size:14px; line-height:1.65; }}
    .panel {{ padding:24px; margin:8px 0 16px; }} .panel h3 {{ margin:0 0 8px; font-size:16px; }} .panel p {{ color:var(--pv-muted); font-size:13px; line-height:1.7; }}
    .result-title {{ display:flex; align-items:center; gap:8px; font-family:Manrope,sans-serif; font-size:30px; font-weight:800; margin:12px 0; color:var(--pv-text); }}
    .result-confidence {{ font-family:Manrope,sans-serif; font-size:clamp(38px,5vw,54px); font-weight:800; letter-spacing:-.055em; margin:5px 0; color:var(--pv-text); }}
    .result-caption {{ color:var(--pv-muted); font-size:12px; margin:16px 0 2px; }}
    .badge {{ background:var(--pv-teal-soft); color:{'#b9e5d8' if dark else '#256b5b'}; font-size:11px; font-weight:700; border-radius:30px; padding:6px 10px; }}
    .badge.warn {{ background:#fff2da; color:#70500d; }}
    .step-card {{ padding:20px; min-height:155px; margin-bottom:12px; }} .step-number {{ color:var(--pv-accent); font-size:12px; font-weight:800; }} .step-card h3 {{ font-size:14px; margin:12px 0 6px; }} .step-card p {{ color:var(--pv-muted); font-size:12px; line-height:1.6; margin:0; }}
    .stButton > button, .stDownloadButton > button {{ border-radius:9px; font-weight:700; font-size:13px; min-height:42px; transition:all .18s ease; }}
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible, [data-testid="stRadio"] label:focus-within {{ outline:3px solid #a99ad4; outline-offset:2px; }}
    .stButton > button[kind="primary"] {{ background:#8271c8; border-color:#8271c8; color:white; }} .stButton > button:hover {{ transform:translateY(-1px); }}
    [data-testid="stFileUploader"] section {{ border:1px dashed #bdb4d9; border-radius:14px; background:var(--pv-soft); padding:25px; }}
    [data-testid="stFileUploader"] section * {{ color:var(--pv-text); }}
    [data-testid="stWidgetLabel"] p {{ color:var(--pv-text); }}
    [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="base-input"] {{ background:var(--pv-surface); color:var(--pv-text); border-color:var(--pv-border); }}
    [data-testid="stButton"] button:not([kind="primary"]), [data-testid="stDownloadButton"] button {{ background:var(--pv-surface); color:var(--pv-text); border-color:var(--pv-border); }}
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] label {{ color:var(--pv-muted); }}
    [data-testid="stMetric"] {{ background:var(--pv-surface); border:1px solid var(--pv-border); border-radius:12px; padding:16px; }}
    [data-testid="stMetricValue"] {{ color:var(--pv-text); }}
    hr {{ border-color:var(--pv-border); }}
    @media(max-width: 900px) {{ [data-testid="stMainBlockContainer"] {{ padding: 2rem 1.5rem 3rem; }} .hero-art {{ opacity:.28; right:-60px; }} .hero-wrap h1, .hero-wrap p {{ position:relative; z-index:2; }} }}
    @media(max-width: 720px) {{ [data-testid="stHorizontalBlock"]:has(.feature-card), [data-testid="stHorizontalBlock"]:has(.metric-card), [data-testid="stHorizontalBlock"]:has(.step-card) {{ flex-wrap:wrap; gap:.65rem; }} [data-testid="stHorizontalBlock"]:has(.feature-card) > [data-testid="column"], [data-testid="stHorizontalBlock"]:has(.metric-card) > [data-testid="column"], [data-testid="stHorizontalBlock"]:has(.step-card) > [data-testid="column"] {{ flex:1 1 175px !important; min-width:0; }} .metric-card {{ min-height:92px; padding:14px; }} .metric-card strong {{ font-size:clamp(17px, 4vw, 23px); }} .feature-card {{ min-height:145px; padding:16px; }} .step-card {{ min-height:130px; }} }}
    @media(max-width: 520px) {{ [data-testid="stMainBlockContainer"] {{ padding:1.25rem 1rem 2rem; }} .hero-wrap {{ padding:25px 20px; min-height:220px; }} .hero-art {{ right:-125px; opacity:.18; }} .hero-wrap h1 {{ font-size:30px; }} .topline {{ font-size:10px; padding-bottom:14px; }} .topline span:last-child {{ display:none; }} .page-intro {{ padding-top:22px; }} .page-intro h1 {{ font-size:27px; }} .section-header {{ align-items:flex-start; gap:8px; flex-direction:column; }} .section-header span {{ font-size:10px; }} .panel {{ padding:18px; }} }}
    </style>
    """, unsafe_allow_html=True)
