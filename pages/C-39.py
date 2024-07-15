import streamlit as st
import os
st.set_page_config(initial_sidebar_state="collapsed")
st.title("C-39")
if st.button("Go Home"):
    st.write("")
    st.switch_page("main.py")

# File to store the notes
FILE_PATH = "pages/notes-39.txt"

def save_note_to_file(note_content):
    with open(FILE_PATH, "w") as file:
        file.write(note_content)

def load_note_from_file():
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r") as file:
            return file.read()
    return ""

# Load existing note
note_content = load_note_from_file()

# Initialize session state
if "text_area_content" not in st.session_state:
    st.session_state.text_area_content = note_content

def save_note():
    save_note_to_file(st.session_state.text_area_content)


with st.container(height=260):
    st.subheader("Basic info")
    with st.expander("click here"):
        with st.container():
            body = "here we can store a lot of information about cows app"
            word = "lorem ipsum" * 1000
            st.write(body + "\n" + word)

    st.markdown("<h1 style='font-size:22px;'>Other info</h1>", unsafe_allow_html=True)
    with st.expander("click here"):
        with st.container():
            body = "here we can store a lot of information about cows app"
            word = "lorem ipsum" * 1000
            st.write(body + "\n" + word)

with st.container(height=300):
    st.subheader("Store some Information")
    st.text_area("Enter your notes here",height=180, on_change=save_note, key="text_area_content")
