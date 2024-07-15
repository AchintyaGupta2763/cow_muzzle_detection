import streamlit as st
st.set_page_config(initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center;font-size:100px; color: red;'>WELCOME</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;font-size:120px; color: red;'>🐮</h1>", unsafe_allow_html=True)

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

up, down = st.columns(2, vertical_alignment="center")

if up.button("CHECK CLASS MANUALLY", use_container_width=True):
    st.write("")
    st.switch_page("pages/pages1.py")
if down.button("USE CAMERA", use_container_width=True):
    st.write("")
    st.switch_page("pages/pages2.py")

