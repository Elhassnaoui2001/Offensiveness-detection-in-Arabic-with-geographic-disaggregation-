# import streamlit as st

# st.set_page_config(
#     page_title="Multipage App",
#     page_icon="👋",
# )

# st.title("Main Page")
# st.sidebar.success("Select a page above.")

# if "my_input" not in st.session_state:
#     st.session_state["my_input"] = ""

# my_input = st.text_input("Input a text here", st.session_state["my_input"])
# submit = st.button("Submit")
# if submit:
#     st.session_state["my_input"] = my_input
#     st.write("You have entered: ", my_input)

import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Dialect Detection and Text Analysis",
    page_icon="🌐",
)

# Introduction
st.title("Dialect Detection and Analysis of Offensive Text")
