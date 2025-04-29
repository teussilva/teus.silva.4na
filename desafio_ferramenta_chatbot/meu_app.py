import streamlit as st

st.title("Meu Primeiro Aplicativo Streamlit")

st.write("Olá! Este é um aplicativo simples criado com Streamlit.")

nome = st.text_input("Digite seu nome:")
if nome:
    st.write(f"Olá, {nome}!")