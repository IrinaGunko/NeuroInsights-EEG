import streamlit as st
from tabs import upload_file, preprocessing, visualization, feature_extraction, ml_prediction

TABS = {
    "Upload EEG File   ": upload_file,
    "Preprocessing     ": preprocessing,
    "EEG Visualization ": visualization,
    "Extract Features  ": feature_extraction,
    "Apply ML Model    ": ml_prediction,
}

def main():
    if "current_file_name" not in st.session_state:
        st.session_state.current_file_name = None

    st.sidebar.header("Current File")
    if st.session_state.current_file_name:
        st.sidebar.info(f"**{st.session_state.current_file_name}**")
    else:
        st.sidebar.warning("No file selected.")

    if "tab" not in st.session_state:
        st.session_state.tab = "Upload EEG File   "

    st.sidebar.title("Navigation")
    for tab_name in TABS.keys():
        if st.sidebar.button(tab_name):
            st.session_state.tab = tab_name

    current_tab = TABS[st.session_state.tab]
    current_tab.render()


if __name__ == "__main__":
    main()
