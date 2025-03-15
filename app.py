import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from frontend.about import about_me
from frontend.main import project
from frontend.home import home_ui


# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Project", "About Me"],
        icons=["house", "app-indicator" ,"person-video3"],
        menu_icon="cast",
        default_index=1,
    )
if selected == "Project":
    project()
if selected == "Home":
    home_ui()

if selected == "About Me":
    about_me()
