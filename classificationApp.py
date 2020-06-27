# Bibliotecas basicas
import streamlit as st
import pandas as pd
import numpy as np

# Sklearn - Geral 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Sklearn - Metricas
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

# Sklearn - Modelos de classificacao
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def main():
	st.title('Aplicativo demonstrativo de modelos de Classificao Binaria')
	st.sidebar.title('Aplicativo demonstrativo de modelos de Classificao Binaria')
	st.markdown('Escolha entre diferentes modelos e teste os parametros')
	st.sidebar.markdown('Escolha entre diferentes modelos e teste os parametros')

	# Definindo funcao para fazer loading dos dados
	# o @st.cache faz o programa manter os dados na memoria
	# Entao, so sera refeito o loading se os dados forem modificados 
	# LabelEncoder transforma os argumentos de string para numericos (cada tipo diferente de string sera um valor numerico diferente) 
	
	@st.cache(persist=True)
	def load_data():
		data = pd.read_csv(r'https://datahub.io/machine-learning/mushroom/r/mushroom.csv')
		labelencoder = LabelEncoder()
		for col in data.columns:
			data[col] = labelencoder.fit_transform(data[col].astype(str))
		return data

	# Funcao para fazer split dos dados (inicialmente setado para 70/30)
	def split(df):
		y = df['class']
		x = df.drop(columns=['class'])
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
		return x_train, x_test, y_train, y_test

	# Funcao que retornara os plots de acordo com as metricas escolhidas
	def plot_metrics(metrics_list):
		if 'Matriz de Confusao' in metrics_list:
			st.subheader('Matriz de Confusao')
			plot_confusion_matrix(model, x_test, y_test, display_labels=class_names,values_format='')
			st.pyplot()
		if 'Curva ROC' in metrics_list:
			st.subheader('Curva ROC')
			plot_roc_curve(model, x_test, y_test)
			st.pyplot()
		if 'Curva de Precisao/Recordacao' in metrics_list:
			st.subheader('Curva de Precisao/Recordacao')
			plot_precision_recall_curve(model, x_test, y_test)
			st.pyplot()

	df = load_data()
	class_names = ['Comestivel', 'Venenoso']
	x_train, x_test, y_train, y_test = split(df)
	
	# Escolha de classificador 
	st.sidebar.subheader('Escolha o classificador:')
	classifier = st.sidebar.selectbox('Classificador', ('SVM', 'Regressao Logistica', 'Floresta Aleatoria'))

	# Ifs para cada classificador
	
	# =====SVM=====
	if classifier == 'SVM':
		st.sidebar.subheader('Hiperparametros do modelo')
		# Escolha dos hiperparametros
		C = st.sidebar.number_input('C (Parametro de regularizacao)', 0.01, 10.0, step=0.01, key='C_SVM')
		kernel = st.sidebar.radio('Kernel', ('rbf', 'linear'), key = 'kernel')
		gamma = st.sidebar.radio('Gamma (Coeficiente do Kernel)', ('scale', 'auto'), key = 'gamma')
	
	
		metrics = st.sidebar.multiselect('Quais metricas plotar?', ('Matriz de Confusao', 'Curva ROC', 'Curva de Precisao/Recordacao'))
		
		if st.sidebar.button('Classificar', key = 'classificar'):
			st.subheader('SVM')
			model = SVC(C=C, kernel=kernel, gamma=gamma)
			model.fit(x_train, y_train)
			accuracy = model.score(x_test, y_test)
			y_pred = model.predict(x_test)
			st.write('Acuracia: ', accuracy.round(2))
			st.write('Precisao: ', precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write('Recordacao: ', recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)

	# =====RegressaoLogistica=====
	if classifier == 'Regressao Logistica':
		st.sidebar.subheader('Hiperparametros do modelo')
		# Escolha dos hiperparametros
		C = st.sidebar.number_input('C (Parametro de regularizacao)', 0.01, 10.0, step=0.01, key='C_LR')
		max_iter = st.sidebar.slider('Numero maximo de iteracoes', 100, 500, key='max_iter')
	
		metrics = st.sidebar.multiselect('Quais metricas plotar?', ('Matriz de Confusao', 'Curva ROC', 'Curva de Precisao/Recordacao'))
		
		if st.sidebar.button('Classificar', key = 'classificar'):
			st.subheader('Regressao Logistica')
			model = LogisticRegression(C=C, max_iter=max_iter)
			model.fit(x_train, y_train)
			accuracy = model.score(x_test, y_test)
			y_pred = model.predict(x_test)
			st.write('Acuracia: ', accuracy.round(2))
			st.write('Precisao: ', precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write('Recordacao: ', recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)

		
	# =====Floresta Aleatoria=====
	if classifier == 'Floresta Aleatoria':
		st.sidebar.subheader('Hiperparametros do modelo')
		# Escolha dos hiperparametros
		n_estimators = st.sidebar.number_input('Numero de arvores na floresta', 100, 5000, step=10, key='n_estimators')
		max_depth = st.sidebar.number_input('Maxima profundidade da arvore', 1, 20, step=1, key='max_depth')
		bootstrap = st.sidebar.radio('Bootstrap das amostras quando criar arvore', ('Verdadeiro', 'Falso'), key = 'boostrap')
	
		metrics = st.sidebar.multiselect('Quais metricas plotar?', ('Matriz de Confusao', 'Curva ROC', 'Curva de Precisao/Recordacao'))
		
		if st.sidebar.button('Classificar', key = 'classificar'):
			st.subheader('Floresta Aleatoria')
			model = RandomForestClassifier(n_estimators=n_estimators, max_depth = max_depth, bootstrap = bootstrap)
			model.fit(x_train, y_train)
			accuracy = model.score(x_test, y_test)
			y_pred = model.predict(x_test)
			st.write('Acuracia: ', accuracy.round(2))
			st.write('Precisao: ', precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write('Recordacao: ', recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)

	# Checkbox para visualizacao dos dados		
	if st.sidebar.checkbox("Ver dados originais", False):
        	st.subheader("Mushroom Data Set (Classificacao)")
        	st.write(df)

	st.markdown("Este [data set](https://archive.ics.uci.edu/ml/datasets/Mushroom) inclui descricoes de amostras hipoteticas correspondentes a 23 especies de cogumelos "
        "das familias Agaricus e Lepiota (pp. 500-525). Cada especie foi catalogada como definitivamente comestivel, definitivamente venenosa, "
        "ou como comestibilidade desconhecida ou nao recomendada. Para este estudo esta ultima foi combinada com a venenosa.")
        	

if __name__ == '__main__':
	main()