import streamlit as st
import pandas as pd
 
st.write("""
Test streamlit dashboard
Hello world!
This is for othello bot deployment
""")


def process_text(input_text):
    # need to link subprocess
    return result

st.title("오델로 기보 입력")

user_input = st.text_input("기보를 입력하세요:")

if st.button("제출"):
    output = process_text(user_input)
    st.write("결과:", output)