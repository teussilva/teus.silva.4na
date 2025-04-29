import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

sia = SentimentIntensityAnalyzer()

st.title('Detecção de Sentimento')
st.subheader('Digite o texto abaixo:')
user_text = st.text_area('Seu texto aqui', height=150, placeholder='Digite aqui uma frase para análise de sentimento...')
botao_analise_sentimento = st.button('Analisar Sentimento')

if botao_analise_sentimento:
    if len(user_text) > 0:
        scores = sia.polarity_scores(user_text)
        compound_score = scores['compound']
        neg_score = scores['neg']

        st.subheader('Resultado da Análise:')
        st.write(f'Pontuação (Composto): {compound_score:.2f}%')
        if compound_score >= 0.05:
            st.write('Sentimento: Positivo')
        elif compound_score <= -0.05:
            st.write('Sentimento: Negativo')
        else:
            st.write('Sentimento: Neutro')
    else:
        st.warning('Por favor, digite algum texto para analisar.')
else:
     st.warning('Nenhum texto inserido, Por favor preencha o campo obrigatório!!!')