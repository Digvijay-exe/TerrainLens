"""Groq-powered road perception research page."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common import render_sidebar_info
from terrainlens.av_perception import (
    DEFAULT_MODEL,
    GroqPerceptionError,
    analyze_road_image,
    decide_driving_action,
    get_groq_api_key,
    render_annotated_image,
)

st.set_page_config(
    page_title="TerrainLens — AV Auto-Route",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)
render_sidebar_info()

st.title("🚘 AV Auto-Route & Road Hazard Perception")
st.caption("Groq vision analysis • localized evidence • conservative route-risk rules")
st.warning(
    "Research advisory only — not a certified autonomous-driving system. "
    "A single RGB image cannot prove a hazard's cause, measure real-world clearance, "
    "or guarantee a safe route. Do not use these outputs to control a real vehicle."
)

with st.sidebar:
    st.markdown("### Vision engine")
    st.code(DEFAULT_MODEL, language="text")
    key_present = bool(get_groq_api_key())
    if key_present:
        st.success("Groq API key detected")
    else:
        st.error("Groq API key not found")
        st.caption("Set GROQ_API_KEY in your environment or .streamlit/secrets.toml.")
    st.caption("Optional override: GROQ_VISION_MODEL")

upload_col, options_col = st.columns([1.65, 1])
with upload_col:
    st.markdown("### 1. Provide a road image")
    uploaded = st.file_uploader(
        "Upload road photograph", type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear forward-facing image with the road visible. Avoid private or sensitive images.",
    )
    camera = st.camera_input("Or capture a frame with your camera")
    source = camera if camera is not None else uploaded
    if source is not None:
        st.image(source, caption="Input road image", use_container_width=True)
with options_col:
    st.markdown("### 2. Analysis settings")
    st.info("Model: Groq vision endpoint\n\nOutput: hazards, normalized boxes, road corridor and uncertainty")
    st.caption("The model's confidence values are subjective estimates, not calibrated probabilities.")
    run = st.button("Analyze road with Groq", type="primary", use_container_width=True, disabled=source is None or not key_present)

if run and source is not None:
    image_bytes = source.getvalue()
    with st.spinner("Sending image to Groq vision model and validating the structured response…"):
        try:
            perception = analyze_road_image(image_bytes)
            decision = decide_driving_action(perception)
            annotated = render_annotated_image(image_bytes, perception)
            st.session_state["av_perception_result"] = {
                "perception": perception,
                "decision": decision,
                "annotated": annotated,
                "original_name": getattr(source, "name", "camera_capture.jpg"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": __import__("os").getenv("GROQ_VISION_MODEL", DEFAULT_MODEL),
            }
        except (GroqPerceptionError, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected analysis error: {exc}")
            st.caption("No driving action was generated. Check the terminal for diagnostic details.")

result = st.session_state.get("av_perception_result")
if result:
    perception = result["perception"]
    decision = result["decision"]
    st.markdown("---")
    st.markdown("### 3. Perception results")
    action = decision["action"]
    if action == "EMERGENCY_STOP":
        st.error(f"🛑 {action} — {decision['reason']}")
    elif action == "REDUCE_SPEED":
        st.warning(f"⚠️ {action} — {decision['reason']}")
    else:
        st.success(f"✓ {action} — {decision['reason']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detected hazards", len(perception["hazards"]))
    c2.metric("Corridor hazards", len(decision["corridor_hazards"]))
    c3.metric("Image risk index", f"{decision['risk_score']}/100")
    c4.metric("Visibility", perception["visibility"].title())

    left, right = st.columns([1.45, 1])
    with left:
        st.markdown("#### Localized evidence")
        st.image(result["annotated"], caption="Boxes show model-localized hazards; green outline shows estimated corridor.", use_container_width=True)
        st.download_button(
            "Download annotated image",
            data=result["annotated"],
            file_name="terrainlens_road_perception.jpg",
            mime="image/jpeg",
        )
    with right:
        st.markdown("#### Scene interpretation")
        st.write(perception["scene_summary"])
        st.write(f"**Surface:** {perception['road_surface']}")
        if perception.get("image_quality_warning"):
            st.warning(perception["image_quality_warning"])
        st.markdown("#### Hazards")
        if not perception["hazards"]:
            st.info("No hazards were returned. This does not prove the road is hazard-free.")
        for i, hazard in enumerate(perception["hazards"], 1):
            with st.expander(f"{i}. {hazard['type'].replace('_', ' ').title()} — {hazard['severity'].upper()}"):
                st.write(hazard["description"] or "No additional description.")
                st.write(f"**Model confidence:** {hazard['confidence']:.2f}")
                st.write(f"**Normalized box:** `{[round(v, 3) for v in hazard['bbox']]}`")
                st.write(f"**Model corridor flag:** {hazard['in_drivable_corridor']}")
    st.markdown("#### Uncertainty and limitations")
    if perception["uncertainties"]:
        for uncertainty in perception["uncertainties"]:
            st.markdown(f"- {uncertainty}")
    else:
        st.write("The model did not list explicit uncertainties; assume important uncertainty remains.")
    st.caption(
        f"Model: {result['model']} · Timestamp: {result['timestamp']} · "
        "Decision rules are deterministic and separate from the model-generated narrative."
    )
    export = {
        "schema_version": "1.0",
        "timestamp_utc": result["timestamp"],
        "model": result["model"],
        "source_image_name": result["original_name"],
        "perception": perception,
        "decision": decision,
        "limitations": [
            "Single-image research advisory; not a certified vehicle controller.",
            "Normalized boxes are not metric distances.",
            "Model confidence is not calibrated.",
            "Raised surface appearance does not prove a subterranean root cause.",
        ],
    }
    st.download_button(
        "Export JSON audit report",
        data=json.dumps(export, indent=2),
        file_name="terrainlens_av_audit.json",
        mime="application/json",
    )
