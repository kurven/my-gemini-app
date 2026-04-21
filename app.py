import streamlit as st
import json
import os
import re
from datetime import datetime, date, time as time_type
import pandas as pd
import plotly.graph_objects as go

# Load .env file if present (local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try importing Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Knit Studio",
    page_icon="🧶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background-color: #faf8f5; }
  [data-testid="stSidebar"] { background-color: #ffffff; }
  .metric-card {
    border-radius: 14px; padding: 14px 16px; height: 100%;
    margin-bottom: 4px;
  }
  .metric-value { font-size: 22px; font-weight: 500; color: #383030; line-height: 1.1; }
  .metric-label { font-size: 11px; font-weight: 500; color: #786868; margin-top: 5px;
    text-transform: uppercase; letter-spacing: 0.06em; }
  .metric-sub { font-size: 11px; color: #b0a0a0; margin-top: 2px; }
  .section-label {
    font-size: 11px; font-weight: 500; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 8px;
  }
  .idea-card {
    background: white; border-radius: 14px; padding: 16px;
    border-top: 1px solid #ede9e3; border-right: 1px solid #ede9e3;
    border-bottom: 1px solid #ede9e3; margin-bottom: 12px;
  }
  div[data-testid="stForm"] { border: none; padding: 0; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
CATEGORIES = [
    "Yarn Haul", "Tutorial", "WIP/Progress", "Pattern",
    "Lifestyle", "Colour Theory", "Q&A", "Unboxing", "Other"
]
HOOKS = ["POV", "Question", "Statement", "Aesthetic", "Value", "Story", "Trend Audio", "Other"]
CAT_COLORS = {
    "Yarn Haul": "#c07890", "Tutorial": "#9a7fcb", "WIP/Progress": "#68a880",
    "Pattern": "#6090c8", "Lifestyle": "#c09840", "Colour Theory": "#c07890",
    "Q&A": "#9a7fcb", "Unboxing": "#68a880", "Other": "#90a0a0",
}
ACCENTS = ["#9a7fcb", "#c07890", "#68a880", "#c09840", "#6090c8", "#4aaa98", "#a890c0"]
DATA_FILE = "knit_studio_data.json"
SAMPLE_POSTS = [
    {"id": 1, "date": "2025-03-12", "title": "Yarn haul from indie dyer", "category": "Yarn Haul",
     "hook": "Aesthetic", "views": 42000, "likes": 3200, "comments": 280, "shares": 410,
     "saves": 890, "hashtags": "#knitter #yarntok", "notes": "_sample"},
    {"id": 2, "date": "2025-03-28", "title": "Colour theory for knitters", "category": "Colour Theory",
     "hook": "Value", "views": 68000, "likes": 5100, "comments": 390, "shares": 720,
     "saves": 1540, "hashtags": "#knittersoftiktok", "notes": "_sample"},
    {"id": 3, "date": "2025-04-10", "title": "Beginner colourwork pattern", "category": "Pattern",
     "hook": "Value", "views": 89000, "likes": 7200, "comments": 560, "shares": 1100,
     "saves": 2300, "hashtags": "#knittinglove", "notes": "_sample"},
]
DEFAULT_HASHTAGS = [
    {"tag": "#knittersoftiktok", "w": 9}, {"tag": "#knitter", "w": 8},
    {"tag": "#knittingaesthetic", "w": 7}, {"tag": "#knittinglove", "w": 7},
    {"tag": "#knittingtiktok", "w": 6}, {"tag": "#yarntok", "w": 5},
    {"tag": "#typebknitter", "w": 5}, {"tag": "#colourfulknitting", "w": 4},
    {"tag": "#beginnerknitter", "w": 4}, {"tag": "#cottagecore", "w": 3},
    {"tag": "#handknit", "w": 3}, {"tag": "#slowfashion", "w": 2},
]

# ─── Data persistence ─────────────────────────────────────────────────────────
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"posts": list(SAMPLE_POSTS), "audience": {}}


def save_data():
    data = {
        "posts": st.session_state.posts,
        "audience": st.session_state.audience,
    }
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass  # File writes won't persist on Streamlit Cloud — use Export instead


def init_state():
    if "posts" not in st.session_state:
        saved = load_data()
        st.session_state.posts = saved.get("posts", list(SAMPLE_POSTS))
        st.session_state.audience = saved.get("audience", {})
    if "audience" not in st.session_state:
        st.session_state.audience = {}

# ─── Gemini ───────────────────────────────────────────────────────────────────
def get_api_key():
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    else:
        return ""

def call_gemini(prompt: str, use_search: bool = False, json_output: bool = False):
    """Call Gemini API. Returns (result_text, error_string)."""
    if not GEMINI_AVAILABLE:
        return None, "Package not installed. Run: pip install google-generativeai"
    api_key = get_api_key()
    if not api_key:
        return None, "No API key found. Add GEMINI_API_KEY to your .env file (locally) or Streamlit Secrets (on cloud)."
    try:
        genai.configure(api_key=api_key)
        config_kwargs = {}
        if json_output:
            config_kwargs["response_mime_type"] = "application/json"
        gen_config = genai.GenerationConfig(**config_kwargs) if config_kwargs else None

        if use_search:
            try:
                model = genai.GenerativeModel(
                    "gemini-1.5-flash",
                    tools=[{"google_search_retrieval": {}}],
                )
                resp = model.generate_content(prompt, generation_config=gen_config)
            except Exception:
                # Fall back to standard model if search grounding fails
                model = genai.GenerativeModel("gemini-1.5-flash")
                resp = model.generate_content(prompt, generation_config=gen_config)
        else:
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt, generation_config=gen_config)

        return resp.text, None
    except Exception as e:
        return None, str(e)

# ─── Utilities ────────────────────────────────────────────────────────────────
def calc_eng(post: dict) -> float:
    views = int(post.get("views") or 0)
    if not views:
        return 0.0
    interactions = sum(int(post.get(k) or 0) for k in ["likes", "comments", "shares", "saves"])
    return round(interactions / views * 100, 2)


def fmt(n) -> str:
    n = float(n or 0)
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(int(n))


def is_sample(post: dict) -> bool:
    return str(post.get("notes", "")).startswith("_sample")


def own_posts() -> list:
    return [p for p in st.session_state.posts if not is_sample(p)]


def avg_eng(posts: list) -> float:
    if not posts:
        return 0.0
    return round(sum(calc_eng(p) for p in posts) / len(posts), 2)

# ─── UI helpers ───────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, sub: str = "", bg: str = "#ede8f6") -> str:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card" style="background:{bg}">
      <div class="metric-value">{value}</div>
      <div class="metric-label">{label}</div>
      {sub_html}
    </div>"""


def section_label(text: str, color: str = "#786868") -> None:
    st.markdown(
        f'<div class="section-label" style="color:{color}">{text}</div>',
        unsafe_allow_html=True,
    )


def word_cloud(tags: list) -> None:
    colors = ["#9a7fcb", "#c07890", "#68a880", "#c09840", "#6090c8", "#4aaa98", "#a890c0"]
    sorted_tags = sorted(tags, key=lambda x: -x["w"])
    spans = [
        f'<span style="font-size:{12 + t["w"]*2.2:.0f}px;color:{colors[i % len(colors)]};'
        f'font-weight:{"500" if t["w"] > 5 else "400"};margin:4px 7px;display:inline-block">'
        f'{t["tag"]}</span>'
        for i, t in enumerate(sorted_tags)
    ]
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;padding:8px 0;line-height:2.2">'
        f'{"".join(spans)}</div>',
        unsafe_allow_html=True,
    )

# ─── Chart helpers ────────────────────────────────────────────────────────────
CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=8, r=8, t=8, b=8),
    font=dict(family="sans-serif", color="#786868", size=11),
    showlegend=False,
)


def area_chart(df: pd.DataFrame, x: str, y: str, color: str = "#9a7fcb"):
    fig = go.Figure(go.Scatter(
        x=df[x], y=df[y], mode="lines+markers", fill="tozeroy",
        line=dict(color=color, width=2),
        fillcolor=color + "26",
        marker=dict(size=5, color=blue),
    ))
    fig.update_layout(**CHART_BASE, height=155)
    fig.update_xaxes(showgrid=True, gridcolor="#ede9e3", showline=False, tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor="#ede9e3", showline=False, tickfont=dict(size=10))
    return fig


def bar_chart(labels: list, values: list, colors: list = None):
    if not colors:
        colors = [ACCENTS[i % len(ACCENTS)] for i in range(len(labels))]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, marker=dict(cornerradius=5)))
    fig.update_layout(**CHART_BASE, height=175)
    fig.update_xaxes(showgrid=False, showline=False, tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor="#ede9e3", showline=False, tickfont=dict(size=10))
    return fig

# ─── Page: Content Log ────────────────────────────────────────────────────────
def page_content():
    st.markdown("### Content log")
    st.caption("Your logged TikTok posts and performance")

    posts = st.session_state.posts
    total_views = sum(int(p.get("views") or 0) for p in posts)
    total_saves = sum(int(p.get("saves") or 0) for p in posts)
    best = max(posts, key=calc_eng, default=None) if posts else None

    # Stat cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(metric_card("Posts logged", str(len(posts)), bg="#ede8f6"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Total views", fmt(total_views), bg="#f7e8ec"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Avg engagement", f"{avg_eng(posts)}%", bg="#e4f0e8"), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("Total saves", fmt(total_saves), bg="#f6f0de"), unsafe_allow_html=True)
    with c5:
        if best:
            st.markdown(
                metric_card("Top eng. rate", f"{calc_eng(best)}%",
                            best.get("title", "")[:24] + "…", bg="#e4eef8"),
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    if len(posts) > 1:
        sorted_posts = sorted(posts, key=lambda p: p["date"])
        col_l, col_r = st.columns(2)
        with col_l:
            with st.container(border=True):
                st.caption("Engagement rate over time")
                df = pd.DataFrame([{"date": p["date"], "eng": calc_eng(p)} for p in sorted_posts])
                st.plotly_chart(area_chart(df, "date", "eng"), use_container_width=True,
                                config={"displayModeBar": False})
        with col_r:
            with st.container(border=True):
                st.caption("Avg engagement by category")
                cat_d = {}
                for p in posts:
                    cat_d.setdefault(p.get("category", "Other"), []).append(calc_eng(p))
                labels = list(cat_d.keys())
                values = [round(sum(v) / len(v), 2) for v in cat_d.values()]
                clrs = [CAT_COLORS.get(l, "#90a0a0") for l in labels]
                st.plotly_chart(
                    bar_chart([l.split("/")[0] for l in labels], values, clrs),
                    use_container_width=True, config={"displayModeBar": False},
                )

    # Add post form
    with st.expander("➕  Log a new post"):
        with st.form("add_post", clear_on_submit=True):
            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                title = st.text_input("Title / description *", placeholder="e.g. How I curate a yarn palette")
            with r1c2:
                post_date = st.date_input("Date posted", value=date.today())
            with r1c3:
                category = st.selectbox("Category", CATEGORIES)

            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                hook = st.selectbox("Hook style", HOOKS)
            with r2c2:
                hashtags = st.text_input("Hashtags", placeholder="#knittersoftiktok #yarntok")
            with r2c3:
                notes = st.text_input("Notes", placeholder="Observations…")

            r3c1, r3c2, r3c3, r3c4, r3c5 = st.columns(5)
            with r3c1: views = st.number_input("Views", min_value=0, value=0)
            with r3c2: likes = st.number_input("Likes", min_value=0, value=0)
            with r3c3: comments = st.number_input("Comments", min_value=0, value=0)
            with r3c4: shares = st.number_input("Shares", min_value=0, value=0)
            with r3c5: saves = st.number_input("Saves", min_value=0, value=0)

            if st.form_submit_button("Log post 🧶", use_container_width=True, type="primary"):
                if title:
                    new_post = {
                        "id": int(datetime.now().timestamp() * 1000),
                        "date": str(post_date), "title": title,
                        "category": category, "hook": hook,
                        "views": int(views), "likes": int(likes),
                        "comments": int(comments), "shares": int(shares),
                        "saves": int(saves), "hashtags": hashtags, "notes": notes,
                    }
                    st.session_state.posts.append(new_post)
                    save_data()
                    st.success("Post logged!")
                    st.rerun()
                else:
                    st.warning("Please enter a title.")

    # Posts table
    with st.container(border=True):
        tc1, tc2 = st.columns([2, 1])
        with tc1:
            st.caption("All posts")
        with tc2:
            sort_by = st.selectbox("Sort by", ["Date", "Views", "Engagement", "Saves"],
                                   label_visibility="collapsed")

        sort_keys = {
            "Date": lambda p: p.get("date", ""),
            "Views": lambda p: int(p.get("views") or 0),
            "Engagement": calc_eng,
            "Saves": lambda p: int(p.get("saves") or 0),
        }
        sorted_posts = sorted(posts, key=sort_keys[sort_by], reverse=True)

        rows = [{
            "Date": p["date"],
            "Title": ("[sample] " if is_sample(p) else "") + p.get("title", ""),
            "Category": p.get("category", ""),
            "Views": fmt(p.get("views", 0)),
            "Saves": fmt(p.get("saves", 0)),
            "Eng %": f"{calc_eng(p)}%",
            "Hook": p.get("hook", ""),
        } for p in sorted_posts]

        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={"Title": st.column_config.TextColumn(width="large")},
            )
        else:
            st.info("No posts yet — log your first one above!")

        # Delete a post
        if posts:
            st.markdown("<br>", unsafe_allow_html=True)
            titles = [f"{p['date']} — {p.get('title','')[:50]}" for p in posts]
            dc1, dc2 = st.columns([3, 1])
            with dc1:
                to_del = st.selectbox("Select post to delete", titles,
                                      label_visibility="collapsed")
            with dc2:
                if st.button("Delete selected", use_container_width=True):
                    idx = titles.index(to_del)
                    st.session_state.posts.pop(idx)
                    save_data()
                    st.rerun()


# ─── Page: Pattern Analysis ───────────────────────────────────────────────────
def page_patterns():
    st.markdown("### Pattern analysis")
    st.caption("What's actually working — runs on your own posts only")

    posts = own_posts()
    if not posts:
        st.info("Log your own posts first. The three sample posts are excluded from analysis.")
        return

    col_l, col_r = st.columns(2)
    with col_l:
        with st.container(border=True):
            st.caption("Hook style performance (avg eng %)")
            hook_d = {}
            for p in posts:
                hook_d.setdefault(p.get("hook", "Other"), []).append(calc_eng(p))
            if hook_d:
                pairs = sorted(
                    [(h, round(sum(v) / len(v), 2)) for h, v in hook_d.items()],
                    key=lambda x: -x[1],
                )
                labels, values = zip(*pairs)
                st.plotly_chart(bar_chart(list(labels), list(values)),
                                use_container_width=True, config={"displayModeBar": False})

    with col_r:
        with st.container(border=True):
            st.caption("Category rankings")
            rows = []
            for cat in CATEGORIES:
                ps = [p for p in posts if p.get("category") == cat]
                if ps:
                    rows.append({
                        "Category": cat,
                        "Avg eng %": avg_eng(ps),
                        "Posts": len(ps),
                        "Total saves": fmt(sum(int(p.get("saves") or 0) for p in ps)),
                    })
            rows.sort(key=lambda r: -r["Avg eng %"])
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("---")
    btn_c, info_c = st.columns([1, 3])
    with btn_c:
        run = st.button("✨  AI pattern insights", use_container_width=True, type="primary")
    with info_c:
        st.caption("Uses Gemini to surface non-obvious patterns. Works best with 5+ posts.")

    if run:
        summary = "\n".join(
            f'"{p["title"]}" | {p.get("category")} | Hook:{p.get("hook")} '
            f'| Views:{p.get("views")} | Eng:{calc_eng(p)}% | Saves:{p.get("saves")}'
            for p in posts
        )
        prompt = f"""Analyse TikTok posts for a type B knitting creator (whimsical colourful knitwear, top creator in niche, experienced strategist).
Focus on non-obvious patterns, saves intelligence, hook performance differences, genuine content gaps.

{summary}

Return JSON:
{{
  "hookInsight": "1-2 sentences about hook patterns",
  "savesInsight": "1-2 sentences about saves drivers",
  "topPatterns": [
    {{"title": "name", "insight": "what you found", "action": "what to do"}}
  ],
  "contentGaps": ["gap 1", "gap 2", "gap 3"]
}}"""

        with st.spinner("Analysing your content…"):
            result, err = call_gemini(prompt, json_output=True)

        if err:
            st.error(f"Analysis failed: {err}")
        elif result:
            try:
                data = json.loads(result)
                c1, c2 = st.columns(2)
                with c1:
                    with st.container(border=True):
                        section_label("Hook pattern", "#9a7fcb")
                        st.write(data.get("hookInsight", ""))
                with c2:
                    with st.container(border=True):
                        section_label("Saves intelligence", "#c09840")
                        st.write(data.get("savesInsight", ""))

                with st.container(border=True):
                    section_label("Key patterns")
                    for i, pat in enumerate(data.get("topPatterns", [])):
                        color = ACCENTS[i % len(ACCENTS)]
                        st.markdown(
                            f'<div style="padding:12px 14px;background:#faf8f5;border-radius:12px;'
                            f'border-left:3px solid {color};margin-bottom:10px">'
                            f'<div style="font-weight:500;font-size:14px;color:#383030;margin-bottom:3px">{pat.get("title","")}</div>'
                            f'<div style="font-size:13px;color:#786868;margin-bottom:5px;line-height:1.5">{pat.get("insight","")}</div>'
                            f'<div style="font-size:13px;color:{color};font-weight:500">→ {pat.get("action","")}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                gaps = data.get("contentGaps", [])
                if gaps:
                    with st.container(border=True):
                        section_label("Content gaps", "#c07890")
                        cols = st.columns(min(len(gaps), 4))
                        for i, gap in enumerate(gaps):
                            with cols[i % len(cols)]:
                                st.markdown(
                                    f'<div style="padding:6px 12px;background:white;border-radius:20px;'
                                    f'font-size:12px;border:1px solid #ede9e3;text-align:center">{gap}</div>',
                                    unsafe_allow_html=True,
                                )
            except json.JSONDecodeError:
                st.write(result)


# ─── Page: Community Pulse ────────────────────────────────────────────────────
def page_pulse():
    st.markdown("### Community pulse")
    st.caption("Live conversations in your niche — Reddit & Instagram")

    platform_map = {
        "All platforms": "Reddit (r/knitting, r/knittingpatterns) and Instagram knitting community",
        "Reddit only": "Reddit (r/knitting, r/knittingpatterns, r/yarnaddicts)",
        "Instagram only": "Instagram knitting community",
    }
    sel_c, btn_c = st.columns([2, 1])
    with sel_c:
        platform_label = st.selectbox("Platform", list(platform_map.keys()),
                                      label_visibility="collapsed")
    with btn_c:
        run = st.button("🌐  Fetch now", use_container_width=True, type="primary")

    if run:
        pd_text = platform_map[platform_label]
        prompt = f"""Search for active, strategically relevant knitting conversations right now on {pd_text}.
Niche: whimsical colourful knitwear, type B knitters, beginner-friendly patterns.
Focus on: what the community requests, is frustrated by, excited about, or gaps a creator could fill.

Return JSON:
{{
  "trends": ["trend 1", "trend 2", "trend 3"],
  "opportunities": ["opportunity 1", "opportunity 2"],
  "conversations": [
    {{
      "platform": "Reddit or Instagram",
      "title": "conversation title",
      "summary": "brief summary",
      "url": "url or N/A",
      "relevance": "why this matters for the creator",
      "sentiment": "positive or neutral or discussion"
    }}
  ]
}}"""

        with st.spinner("Scanning communities…"):
            result, err = call_gemini(prompt, use_search=True, json_output=True)

        if err:
            st.error(f"Search failed: {err}")
        elif result:
            try:
                data = json.loads(result)
                sentiment_colors = {"positive": "#68a880", "discussion": "#c09840", "neutral": "#6090c8"}

                if data.get("trends"):
                    with st.container(border=True):
                        section_label("Trending now", "#9a7fcb")
                        tags_html = "".join(
                            f'<span style="padding:5px 14px;background:white;border-radius:20px;'
                            f'font-size:12px;border:1px solid #ede9e3;margin:3px;display:inline-block">{t}</span>'
                            for t in data["trends"]
                        )
                        st.markdown(f'<div style="display:flex;flex-wrap:wrap">{tags_html}</div>',
                                    unsafe_allow_html=True)

                if data.get("opportunities"):
                    with st.container(border=True):
                        section_label("Creator opportunities", "#68a880")
                        for opp in data["opportunities"]:
                            st.markdown(f"→ {opp}")

                if data.get("conversations"):
                    st.markdown("**Active conversations**")
                    for conv in data["conversations"]:
                        sc = sentiment_colors.get(conv.get("sentiment", "neutral"), "#6090c8")
                        with st.container(border=True):
                            main_c, link_c = st.columns([5, 1])
                            with main_c:
                                st.markdown(
                                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                                    f'<span style="padding:2px 8px;border-radius:20px;font-size:11px;font-weight:500;'
                                    f'background:{sc}30;color:{sc}">{conv.get("platform","")}</span>'
                                    f'<span style="width:7px;height:7px;border-radius:50%;background:{sc};display:inline-block"></span>'
                                    f'</div>'
                                    f'<div style="font-weight:500;font-size:14px;color:#383030;margin-bottom:4px">{conv.get("title","")}</div>'
                                    f'<div style="font-size:13px;color:#786868;line-height:1.5;margin-bottom:5px">{conv.get("summary","")}</div>'
                                    f'<div style="font-size:12px;color:{sc};font-weight:500">Why it matters: {conv.get("relevance","")}</div>',
                                    unsafe_allow_html=True,
                                )
                            with link_c:
                                url = conv.get("url", "N/A")
                                if url and url != "N/A":
                                    st.link_button("View →", url)
            except json.JSONDecodeError:
                st.write(result)


# ─── Page: Audience ───────────────────────────────────────────────────────────
def page_audience():
    st.markdown("### Audience profile")
    st.caption("Fill in from TikTok Analytics — save when done")

    aud = st.session_state.audience

    c1, c2, c3 = st.columns(3)
    with c1:
        followers = st.number_input("Followers", min_value=0,
                                     value=int(aud.get("followers") or 0), step=100)
        st.session_state.audience["followers"] = followers
    with c2:
        avg_views = st.number_input("Avg views / video", min_value=0,
                                     value=int(aud.get("avgViews") or 0), step=100)
        st.session_state.audience["avgViews"] = avg_views
    with c3:
        profile_visits = st.number_input("Profile visits / month", min_value=0,
                                          value=int(aud.get("profileVisits") or 0), step=100)
        st.session_state.audience["profileVisits"] = profile_visits

    st.markdown("---")
    col_age, col_geo = st.columns(2)

    with col_age:
        with st.container(border=True):
            st.caption("Age distribution (%)")
            defaults_age = {"18–24": 32, "25–34": 41, "35–44": 18, "45+": 9}
            for age_g, default_v in defaults_age.items():
                key = f"age_{age_g}"
                val = st.slider(age_g, 0, 100, int(aud.get(key, default_v)))
                st.session_state.audience[key] = val

            st.markdown("---")
            st.caption("Female audience")
            fem = st.slider("Female %", 0, 100,
                             int(aud.get("femPct", 87)), label_visibility="collapsed")
            st.session_state.audience["femPct"] = fem
            st.progress(fem / 100, text=f"{fem}% female")

    with col_geo:
        with st.container(border=True):
            st.caption("Top countries")
            defaults_geo = [("UK", 34), ("US", 28), ("Australia", 14), ("Sweden", 10), ("Germany", 6)]
            for i, (dc, dp) in enumerate(defaults_geo, 1):
                gc1, gc2 = st.columns([2, 1])
                with gc1:
                    cname = st.text_input(f"Country {i}",
                                          value=aud.get(f"c{i}", dc),
                                          label_visibility="collapsed",
                                          key=f"cname_{i}")
                with gc2:
                    cpct = st.number_input("%", 0, 100,
                                           int(aud.get(f"p{i}", dp)),
                                           label_visibility="collapsed",
                                           key=f"cpct_{i}")
                st.session_state.audience[f"c{i}"] = cname
                st.session_state.audience[f"p{i}"] = cpct

            st.markdown("---")
            st.caption("Peak posting window")
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            pc1, pc2 = st.columns(2)
            with pc1:
                day = st.selectbox("Day", days,
                                    index=days.index(aud.get("peakDay", "Sunday")))
                st.session_state.audience["peakDay"] = day
            with pc2:
                stored_time = aud.get("peakTime", "19:00")
                try:
                    h, m = stored_time.split(":")
                    t_val = time_type(int(h), int(m))
                except Exception:
                    t_val = time_type(19, 0)
                peak_time = st.time_input("Time", value=t_val, label_visibility="collapsed")
                st.session_state.audience["peakTime"] = str(peak_time)[:5]

    notes = st.text_area(
        "Additional notes",
        value=aud.get("notes", ""),
        placeholder="e.g. High save rate on colour theory, international enthusiasts…",
    )
    st.session_state.audience["notes"] = notes

    if st.button("Save audience data", type="primary"):
        save_data()
        st.success("Saved!")


# ─── Page: Sponsor Kit ────────────────────────────────────────────────────────
def page_sponsor():
    st.markdown("### Sponsor & collab kit")
    st.caption("Data-led assets for brand outreach")

    posts = own_posts() or st.session_state.posts
    total_views = sum(int(p.get("views") or 0) for p in posts)
    total_saves = sum(int(p.get("saves") or 0) for p in posts)
    avg_v = round(total_views / len(posts)) if posts else 0
    best = max(posts, key=calc_eng, default=None)
    avg_e = avg_eng(posts)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(metric_card("Avg engagement", f"{avg_e}%", "Industry avg: 1–3%", "#ede8f6"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Avg views / video", fmt(avg_v), f"{len(posts)} videos", "#f7e8ec"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Total saves", fmt(total_saves), "Purchase-intent signal", "#e4f0e8"), unsafe_allow_html=True)
    with c4:
        if best:
            st.markdown(metric_card("Best eng. rate", f"{calc_eng(best)}%",
                                    best.get("title", "")[:22] + "…", "#f6f0de"), unsafe_allow_html=True)

    st.markdown("---")

    if st.button("✨  Generate pitch copy", type="primary"):
        prompt = f"""Write a sponsor pitch for a TikTok creator:
- Niche: Type B knitters — whimsical colourful knitwear, colour curation, beginner patterns
- Position: Top creator in this niche
- Avg engagement: {avg_e}% (industry avg 1-3%)
- Avg views per video: {fmt(avg_v)}
- Total saves: {fmt(total_saves)} across {len(posts)} videos
- Best eng. rate: {f"{calc_eng(best)}%" if best else "N/A"}
- Audience: predominantly women, passionate hobbyists, high spend on craft

Return JSON:
{{
  "oneLiner": "one compelling sentence",
  "emailIntro": "2-3 sentence email intro",
  "whyNow": "why partner now (2-3 sentences)",
  "audienceSnapshot": "audience for sponsors (2-3 sentences)"
}}"""

        with st.spinner("Generating your pitch…"):
            result, err = call_gemini(prompt, json_output=True)

        if err:
            st.error(f"Failed: {err}")
        elif result:
            try:
                st.session_state.pitch = json.loads(result)
            except json.JSONDecodeError:
                st.write(result)

    if "pitch" in st.session_state and st.session_state.pitch:
        pitch = st.session_state.pitch

        with st.container(border=True):
            section_label("One-liner", "#9a7fcb")
            st.markdown(
                f'<div style="font-size:15px;color:#383030;font-style:italic;line-height:1.6;margin-bottom:8px">'
                f'"{pitch.get("oneLiner","")}"</div>',
                unsafe_allow_html=True,
            )
            st.code(pitch.get("oneLiner", ""), language=None)

        for key, label in [("emailIntro", "Email intro"), ("whyNow", "Why work with me now"),
                            ("audienceSnapshot", "Audience snapshot")]:
            with st.container(border=True):
                section_label(label)
                st.write(pitch.get(key, ""))
                st.code(pitch.get(key, ""), language=None)

    st.markdown("---")
    with st.container(border=True):
        st.caption("Your hashtag universe")
        word_cloud(DEFAULT_HASHTAGS)


# ─── Page: Content Ideas ──────────────────────────────────────────────────────
def page_ideas():
    st.markdown("### Content ideas")
    st.caption("AI-generated ideas based on your top posts and live trends")

    posts = own_posts() or st.session_state.posts
    focus_options = [
        "Balanced mix", "Maximise saves", "Maximise engagement",
        "New audience growth", "Trend-led only",
    ]

    fc, bc = st.columns([2, 1])
    with fc:
        focus = st.selectbox("Focus", focus_options, label_visibility="collapsed")
    with bc:
        generate = st.button("✦  Generate ideas", type="primary", use_container_width=True)

    if generate:
        top = sorted(posts, key=calc_eng, reverse=True)[:5]
        top_summary = "\n".join(
            f'"{p["title"]}" | {p.get("category")} | Hook:{p.get("hook")} '
            f'| Eng:{calc_eng(p)}% | Saves:{p.get("saves")}'
            for p in top
        )
        prompt = f"""Generate 6 TikTok ideas for a type B knitting creator (whimsical colourful knitwear, beginner patterns, colour curation). Top creator in niche.

Focus: {focus}
Top performing posts:
{top_summary}

Search for current knitting trends. Ideas must be specific — each needs a clear hook and unique angle. No generic titles.

Return JSON:
{{
  "trendContext": "1-2 sentences on current knitting trends",
  "ideas": [
    {{
      "title": "specific video title",
      "hook": "hook type",
      "category": "category",
      "whyItWorks": "1-2 sentences",
      "formatTip": "brief execution tip",
      "suggestedHashtags": ["#tag1", "#tag2"]
    }}
  ]
}}"""

        with st.spinner("Searching trends and generating ideas…"):
            result, err = call_gemini(prompt, use_search=True, json_output=True)

        if err:
            st.error(f"Failed: {err}")
        elif result:
            try:
                st.session_state.ideas_data = json.loads(result)
            except json.JSONDecodeError:
                st.write(result)

    if "ideas_data" in st.session_state and st.session_state.ideas_data:
        data = st.session_state.ideas_data

        if data.get("trendContext"):
            with st.container(border=True):
                section_label("Trend context", "#4aaa98")
                st.write(data["trendContext"])

        ideas = data.get("ideas", [])
        for i in range(0, len(ideas), 2):
            c1, c2 = st.columns(2)
            for col, idx in [(c1, i), (c2, i + 1)]:
                if idx < len(ideas):
                    idea = ideas[idx]
                    color = ACCENTS[idx % len(ACCENTS)]
                    cat_color = CAT_COLORS.get(idea.get("category", "Other"), "#90a0a0")
                    hashtags_html = "".join(
                        f'<span style="font-size:10px;color:#b0a0a0;background:#faf8f5;'
                        f'padding:2px 6px;border-radius:8px;border:1px solid #ede9e3;margin:2px">{h}</span>'
                        for h in idea.get("suggestedHashtags", [])
                    )
                    with col:
                        st.markdown(
                            f'<div class="idea-card" style="border-left:3px solid {color}">'
                            f'<div style="display:flex;justify-content:space-between;margin-bottom:8px">'
                            f'<span style="padding:2px 8px;border-radius:20px;font-size:11px;font-weight:500;'
                            f'background:{cat_color}30;color:{cat_color}">{idea.get("category","")}</span>'
                            f'<span style="padding:2px 8px;border-radius:20px;font-size:11px;font-weight:500;'
                            f'background:#6090c830;color:#6090c8">{idea.get("hook","")}</span>'
                            f'</div>'
                            f'<div style="font-weight:500;font-size:14px;color:#383030;margin-bottom:6px;line-height:1.4">{idea.get("title","")}</div>'
                            f'<div style="font-size:12px;color:#786868;line-height:1.5;margin-bottom:8px">{idea.get("whyItWorks","")}</div>'
                            f'<div style="font-size:11px;color:{color};font-weight:500;margin-bottom:8px">Format: {idea.get("formatTip","")}</div>'
                            f'<div style="display:flex;flex-wrap:wrap;gap:4px">{hashtags_html}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )


# ─── Page: Hashtag Tracker ────────────────────────────────────────────────────
def page_hashtags():
    st.markdown("### Hashtag tracker")
    st.caption("Performance of hashtags used in your logged posts")

    posts = own_posts() or st.session_state.posts

    # Parse hashtags
    tag_map: dict = {}
    for p in posts:
        raw_tags = re.split(r"\s+", p.get("hashtags", "") or "")
        tags = [t.lower() for t in raw_tags if t.startswith("#")]
        for tag in tags:
            if tag not in tag_map:
                tag_map[tag] = {"tag": tag, "uses": 0, "total_eng": 0.0,
                                "total_views": 0, "total_saves": 0}
            tag_map[tag]["uses"] += 1
            tag_map[tag]["total_eng"] += calc_eng(p)
            tag_map[tag]["total_views"] += int(p.get("views") or 0)
            tag_map[tag]["total_saves"] += int(p.get("saves") or 0)

    if not tag_map:
        st.info("Log posts with hashtags to see data here. "
                "Enter hashtags in the Content log: #knittersoftiktok #yarntok")
        return

    tag_list = []
    for t in tag_map.values():
        n = t["uses"]
        tag_list.append({
            **t,
            "avg_eng": round(t["total_eng"] / n, 2),
            "avg_views": round(t["total_views"] / n),
            "avg_saves": round(t["total_saves"] / n),
        })

    sc, _ = st.columns([1, 3])
    with sc:
        sort_by = st.selectbox(
            "Sort by", ["Avg engagement", "Avg views", "Avg saves", "Uses"],
            label_visibility="collapsed",
        )
    sort_key_map = {"Avg engagement": "avg_eng", "Avg views": "avg_views",
                    "Avg saves": "avg_saves", "Uses": "uses"}
    tag_list.sort(key=lambda x: -x[sort_key_map[sort_by]])

    max_eng = max(t["avg_eng"] for t in tag_list) or 1

    # Performance word cloud
    with st.container(border=True):
        st.caption("Performance cloud — size = avg engagement rate")
        cloud_tags = [
            {"tag": t["tag"], "w": max(1, round((t["avg_eng"] / max_eng) * 9))}
            for t in tag_list
        ]
        word_cloud(cloud_tags)

    # Top 4 cards
    top4 = tag_list[:4]
    bg_colors = ["#ede8f6", "#e4f0e8", "#f6f0de", "#f7e8ec"]
    if top4:
        cols = st.columns(len(top4))
        for col, t, bg in zip(cols, top4, bg_colors):
            with col:
                st.markdown(
                    metric_card(f"Used {t['uses']}×", f"{t['avg_eng']}%", t["tag"], bg),
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        df = pd.DataFrame([{
            "Hashtag": t["tag"],
            "Uses": t["uses"],
            "Avg eng %": t["avg_eng"],
            "Avg views": fmt(t["avg_views"]),
            "Avg saves": fmt(t["avg_saves"]),
        } for t in tag_list])
        st.dataframe(df, hide_index=True, use_container_width=True)

    with st.container(border=True):
        section_label("How to use this", "#9a7fcb")
        st.write(
            "Hashtags correlate with performance but don't cause it. "
            "Look for which tag combinations appear in your strongest posts "
            "rather than chasing individual tags. Focus on your top 3–5 patterns."
        )


# ─── Sidebar + Main ───────────────────────────────────────────────────────────
def main():
    init_state()

    PAGES = {
        "🎬  Content log": page_content,
        "🔍  Pattern analysis": page_patterns,
        "🌐  Community pulse": page_pulse,
        "👥  Audience": page_audience,
        "💼  Sponsor kit": page_sponsor,
        "✦   Content ideas": page_ideas,
        "#   Hashtag tracker": page_hashtags,
    }

    with st.sidebar:
        st.markdown("## 🧶 Knit Studio")
        st.caption("TikTok creator dashboard")
        st.markdown("---")

        page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

        st.markdown("---")

        # API key status
        api_key = get_api_key()
        if api_key:
            st.success("Gemini API connected", icon="✅")
        else:
            st.warning("Add your GEMINI_API_KEY to enable AI features", icon="🔑")

        st.markdown("---")

        # Quick stats
        my = own_posts()
        st.metric("Your posts", len(my))
        if my:
            st.metric("Avg engagement", f"{avg_eng(my)}%")

        st.markdown("---")

        # Export / Import
        st.caption("Your data")
        export_json = json.dumps(
            {"posts": st.session_state.posts, "audience": st.session_state.audience},
            indent=2,
        )
        st.download_button(
            "⬇️  Export data",
            data=export_json,
            file_name="knit_studio_backup.json",
            mime="application/json",
            use_container_width=True,
        )
        uploaded = st.file_uploader("Import data", type="json", label_visibility="collapsed")
        if uploaded:
            try:
                imported = json.load(uploaded)
                st.session_state.posts = imported.get("posts", [])
                st.session_state.audience = imported.get("audience", {})
                save_data()
                st.success("Data imported!")
                st.rerun()
            except Exception:
                st.error("Invalid file — please upload a valid backup.")

    PAGES[page]()


if __name__ == "__main__":
    main()
