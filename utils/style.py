
from pathlib import Path
import streamlit as st


def apply_custom_styles():
    css_file = (
        Path(__file__).resolve().parent.parent
        / "assets"
        / "style.css"
    )

    if css_file.exists():
        css = css_file.read_text(encoding="utf-8")
        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True,
        )
