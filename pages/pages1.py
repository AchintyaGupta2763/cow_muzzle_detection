import streamlit as st
st.set_page_config(initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center;font-size:40px; color: red;'>CHOOSE BELOW</h1>", unsafe_allow_html=True)

if st.button('GO HOME', use_container_width=True):
    st.switch_page("main.py")


buttons_per_row = 5
total_buttons = 50

for i in range(0, total_buttons, buttons_per_row):
    with st.container():
        columns = st.columns(buttons_per_row, gap="small")
        for j in range(buttons_per_row):
            button_index = i + j + 1
            if button_index > total_buttons:
                break
            if columns[j].button(f"C-{button_index:02d}", use_container_width=True):
                st.switch_page(f"pages/C-{button_index:02d}.py")