import pandas as pd
import numpy as np

import nltk
import spacy
import sklearn

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


from gensim.corpora import Dictionary
from gensim.models import LdaModel

nltk.download('stopwords')
nltk.download('punkt')

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import re
import contractions

from nltk.corpus import stopwords
from nltk import word_tokenize
# Used in Lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer 

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import nltk
nltk.download('wordnet')

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

import plotly.express as px

from PIL import Image

# Load the CSV file into a DataFrame
df1 = pd.read_csv(r'C:/Users/sebas/OneDrive/Desktop/ICD/Primera parte/scopus (1).csv')

def clean_text(text_string, punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~'''):
    # Expandir contracciones
    expanded_text = contractions.fix(text_string)
    
    # Eliminar URLs
    no_urls = re.sub(r'https?://\S+|www\.\S+', '', expanded_text)
    
    # Eliminar elementos HTML
    no_html = re.sub(r'<.*?>', '', no_urls)
    
    # Eliminar puntuaciones
    no_punctuations = re.sub(r'[^\w\s]', '', no_html)
    
    # Convertir a minúsculas
    lower_text = no_punctuations.lower()
    
    # Tokenización
    tokens = word_tokenize(lower_text)
    
    # Eliminar números
    tokens = [word for word in tokens if word.isalpha()]
    
    # Cargar y fusionar listas de stopwords
    customlist = ['not', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
                  "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                  "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
                  "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'use', 'howev',
                  'focu', 'year', 'show', 'fase', 'along', 'ieee', 'franci', 'taylor', "new", "also",
                  "autor", "paper", "studi", "author", "right", "develop", "big", "data", "use", "provid",
                  "chang","hr","need","base","elsevi","ltd","springer","model","inform","system","result",
                  "sens","includ","find","person","reserv","approach","process","desing","one","five",
                  "trait","inform","applic","system","analysi","challeng","human","person","base","includ","structur",
                  "develop","human","index","hdi",'develop','human','index','develo']
    stop_words = set(stopwords.words('english')).union(customlist)
    
    # Filtrar stopwords
    filtered_words = [word for word in tokens if word not in stop_words]
    
    # Lematización
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]
    
    # Resultado final
    return ' '.join(stemmed_words)


#converts all the values in a specific columns of the DataFrame data to strings 
df1["Abstract"] = df1["Abstract"].astype(str) 

#Applying a Text Cleaning Function
df1['clean_Abstract'] = df1['Abstract'].apply(clean_text)

stopwords_abs = []  # Initialize the list

# Adding some of the words to the stopwords list 
stopwords_to_remove = ['use',"use",'show','studi','comput','intellig',"right",'gener','integr','howev','focu',
                       'year','show','fase','along','ieee','franci','taylor',"uk","model","inform","system","result",
                       "sens","includ","find","person","reserv","chang","approach","process","desing","one","five","trait",
                       "inform","applic","system","analysi","challeng","human","person","base","includ","structur","human","index","hdi"]

stopwords_abs.extend(stopwords_to_remove)  # Now you can extend it


############# BoW 

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the 'clean_Abstract' column
X = vectorizer.fit_transform(df1['clean_Abstract'])

# Convert the BoW array into a DataFrame
bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())


######### bigrams

# Initialize CountVectorizer with bigram parameter
vectorizer = CountVectorizer(ngram_range=(2, 2))  # For bigrams

# Fit and transform the cleaned abstracts
X = vectorizer.fit_transform(df1['clean_Abstract'])

# Convert to array and then to DataFrame for better visualization
ngram_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())



#########################################


# Título del proyecto
st.title('Thematic Relationships in Academic Publications on the Concept of the Human Development Index')

st.subheader('Bibliometricl Analysis')

st.markdown(
    """
    <p style="font-size:16px; color:gray;">
        For the bibliometric analysis, structured data were used, including publication dates, authors, titles, and more.
        In the first graph, the user can see the most use keywords in the abstracts, 
        with the slide bar can adjust the graph for specific time frames
    </p>
    """,
    unsafe_allow_html=True
)


# Obtener valores únicos de 'Document Type'
document_types = df1['Document Type'].unique()


with st.expander("Filters", expanded=True):
    # Multiselect para seleccionar tipos de documentos con opciones desplegables
    selected_types = st.multiselect(
        "Select Document Types:",
        options=document_types,
        default=document_types  # Selecciona todos por defecto
    )



    # Slider para filtrar por rango de años
    min_year = int(df1['Year'].min())
    max_year = int(df1['Year'].max())

    selected_years = st.slider(
        "Select Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)  # Rango completo por defecto
    )


# Filtrar DataFrame según selección de tipos y años
filtered_df = df1[
    (df1['Document Type'].isin(selected_types)) &
    (df1['Year'] >= selected_years[0]) & 
    (df1['Year'] <= selected_years[1])
]




# Cargar y mostrar una imagen con Pillow
imagen = Image.open(r"C:/Users/sebas/OneDrive/Desktop/EntregaFinal/overlayvisualization_hdi.png")

with st.expander("Occurrences of Human Development", expanded=True):
    st.image(imagen, caption="Visualización del HDI", use_container_width=True)


################### Filtros arriba

st.markdown(
    """
    <p style="font-size:16px; color:gray;">
        In the first graph, two columns from the dataset (keywords and year of publication) were concatenated.
        The results show a steady increase in the use of the terms "human development index" and "HDI".
    </p>
    """,
    unsafe_allow_html=True
)

# Asegurarnos de que los datos están en formato adecuado
filtered_df['Author Keywords'] = filtered_df['Author Keywords'].fillna('')  # Rellenar valores nulos en 'Author Keywords'

# Separar palabras clave y convertirlas a minúsculas
df_keywords = filtered_df['Author Keywords'].str.split('; ')
keywords_exploded = df_keywords.explode().str.lower()  # Definir keywords_exploded

# Lista de palabras clave a excluir
keywords_to_exclude = [
    'human development index',
    'human development',
    'development',
    'hdi',
    'human development index (hdi)'
]

# Filtrar las palabras clave a excluir
keywords_exploded_filtered = keywords_exploded[~keywords_exploded.isin(keywords_to_exclude)]

# Contar las palabras clave restantes
word_counts = keywords_exploded_filtered.value_counts().reset_index()
word_counts.columns = ['Keyword', 'Count']

# Filtrar las 25 palabras clave con mayor frecuencia
word_counts_filtered = word_counts.nlargest(25, 'Count').iloc[1:] 

# Crear gráfico de palabras clave (barras verticales con resultados mayores a la derecha)
fig2 = px.bar(
    word_counts_filtered,
    x='Keyword',  # Palabras clave en el eje x
    y='Count',  # Conteo en el eje y
    title='Keyword Frequency (Excluding Specific Keywords)',
    labels={'Keyword': 'Keywords', 'Count': 'Count'},
    text='Count',
    template='plotly_white'   
)

# Personalización del gráfico
fig2.update_traces(textposition='outside')
fig2.update_layout(
    xaxis=dict(categoryorder='total ascending'),  # Ordenar palabras clave en orden ascendente por frecuencia
    height=800,  # Altura del gráfico
    width=1200,  # Ancho del gráfico
    margin=dict(l=50, r=50, t=50, b=200),  # Ajustar márgenes para etiquetas largas
    xaxis_tickangle=45  # Rotar etiquetas del eje x
)


with st.expander("Keyword Frequency Chart", expanded=True):
    st.plotly_chart(fig2, use_container_width=True)

##################### ocurrencia de key words arriba


# Asegurarnos de que los datos están en formato adecuado
df1['Year'] = pd.to_numeric(df1['Year'], errors='coerce')  # Convertir años a numérico si no lo están
df1['Author Keywords'] = df1['Author Keywords'].fillna('')  # Rellenar valores nulos en 'Author Keywords'

# Lista de palabras clave que se quieren contar
keywords_to_count = [
    'human development index',
    'human development',
    'hdi',
    'human development index (hdi)'
]

# Separar palabras clave y convertirlas a minúsculas
df_keywords = df1['Author Keywords'].str.split('; ')
keywords_exploded = df_keywords.explode().str.lower()

# Filtrar por las palabras clave especificadas y mantener los índices originales
filtered_keywords_indices = keywords_exploded[keywords_exploded.isin(keywords_to_count)].index

# Usar los índices para filtrar el DataFrame original
filtered_keywords = df1.loc[filtered_keywords_indices]

# Contar apariciones por año
keyword_counts_by_year = (
    filtered_keywords.groupby('Year').size().reset_index(name='Count')
)

# Crear gráfico
fig = px.bar(
    keyword_counts_by_year,
    x='Year',
    y='Count',
    title='Occurrences of Human Development-related Keywords by Year',
    labels={'Year': 'Year', 'Count': 'Occurrences'},
    template='plotly_white'
)

# Ajustar tamaño del gráfico
fig.update_layout(
    autosize=True,
    height=600,
    width=800,
    margin=dict(l=20, r=20, t=50, b=50),
)

with st.expander("Occurrences of Human Development", expanded=True):
    st.plotly_chart(fig, use_container_width=True)


############### hdi keyword arriba

# Agrupar y sumar las citas por revista
journal_citations = df1.groupby('Source title')['Cited by'].sum().sort_values(ascending=False).head(10)

# Convertir a DataFrame para usar en Plotly
journal_citations_df = journal_citations.reset_index()
journal_citations_df.columns = ['Journal', 'Total Citations']

# Ordenar los datos de manera ascendente (opcional si ya está ordenado)
journal_citations_df = journal_citations_df.sort_values(by='Total Citations', ascending=True)

# Crear gráfico interactivo con Plotly
fig = px.bar(
    journal_citations_df,
    x='Total Citations',
    y='Journal',
    text='Total Citations',
    title='Top 10 Most Cited Journals',
    labels={'Journal': 'Journal Title', 'Total Citations': 'Citations'},
    template='plotly_white',
    
)

# Personalizar el gráfico
fig.update_traces(textposition='outside')  # Mostrar los valores fuera de las barras
fig.update_layout(
    height=600,          # Ajustar altura del gráfico
    margin=dict(l=40, r=40, t=40, b=80)  # Ajustar márgenes
)

# Usar un expander para mostrar el gráfico
with st.expander("Top 10 Most Cited Journals"):
    st.plotly_chart(fig)


######################### Titulos de revista mas citados arriba

# Asegurarnos de que la columna 'Cited by' sea numérica
df1['Cited by'] = pd.to_numeric(df1['Cited by'], errors='coerce').fillna(0).astype(int)

# Filtrar los datos según el rango de años seleccionado
filtered_df = df1[
    (df1['Document Type'].isin(selected_types)) & 
    (df1['Year'] >= selected_years[0]) & 
    (df1['Year'] <= selected_years[1])
]

# Agrupar por título y sumar el número de citas
top_articles_grouped = (
    filtered_df.groupby('Title')['Cited by']
    .sum()
    .reset_index()
    .rename(columns={'Cited by': 'Total Citations'})
)

# Seleccionar los 10 artículos más citados
top_articles = top_articles_grouped.nlargest(10, 'Total Citations')

# Recortar los nombres de los títulos a las primeras 6 palabras
top_articles['Title'] = top_articles['Title'].apply(lambda x: ' '.join(x.split()[:14]))


# Crear gráfico de barras verticales
fig5 = px.bar(
    top_articles,
    x='Title',             # Títulos de los artículos en el eje X
    y='Total Citations',   # Conteo de citas en el eje Y
    title='Top 10 Most Cited Articles by Selected Year Range',
    labels={'Title': 'Article Title', 'Total Citations': 'Citations'},
    text='Total Citations',  # Mostrar el conteo como texto en las barras
    template='plotly_white'
)

# Personalización del gráfico
fig5.update_traces(textposition='outside')  # Posicionar texto fuera de las barras
fig5.update_layout(
    xaxis=dict(
        categoryorder='total ascending',  # Ordenar categorías de menor a mayor
        title=None,                       # Remover título del eje X para más limpieza
        tickangle=45,                     # Rotar etiquetas para mejor legibilidad
        tickfont=dict(size=8)            # Reducir tamaño de las etiquetas
    ),
    yaxis=dict(
        title=None                        # Remover título del eje Y para más limpieza
    ),
    margin=dict(l=50, r=20, t=50, b=100),  # Ajustar márgenes
    height=700                             # Altura del gráfico
)

with st.expander("Top Cited Articles", expanded=True):
    st.plotly_chart(fig5, use_container_width=True, key="top_cited_articles_chart")


################## Articulos mas citados arriba 





#################################


st.subheader('Content analysis')


###############################
# Crear WordCloud dentro de un expander
with st.expander("WordCloud Analysis", expanded=True):
    # Configuración del WordCloud
    plt.subplots(figsize=(16, 13))
    wordcloud = WordCloud(
        background_color='white', max_words=10000, width=1500,
        stopwords=stopwords_abs, height=1080
    ).generate(" ".join(df1['clean_Abstract']))
    
    # Título del WordCloud
    plt.title("Most common abstract words (without stopwords and some additional terms)", fontsize=20)
    
    # Mostrar el WordCloud
    plt.imshow(wordcloud.recolor(colormap='viridis'))
    plt.axis('off')
    st.pyplot(plt)



#####################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Expander para la sección de WordCloud con TF-IDF
with st.expander("TF-IDF WordCloud", expanded=True):
    # Inicializar TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Ajustar y transformar los abstracts limpios
    tfidf_matrix = tfidf_vectorizer.fit_transform(df1['clean_Abstract'])

    # Convertir la matriz TF-IDF en DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), 
        columns=tfidf_vectorizer.get_feature_names_out()
    )

    # Sumar los valores TF-IDF para cada palabra
    sum_tfidf = tfidf_df.sum(axis=0)

    # Ordenar y seleccionar las 10 palabras con mayor TF-IDF
    sorted_tfidf = sum_tfidf.sort_values(ascending=False)[:10]

    # Generar el WordCloud
    wordcloud = WordCloud(
        width=800, height=400, background_color='white'
    ).generate_from_frequencies(sum_tfidf)

    # Crear el gráfico del WordCloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("TF-IDF Wordcloud", fontsize=20)
    
    # Mostrar el WordCloud en Streamlit
    st.pyplot(plt)

#################################

import streamlit as st
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models import LdaModel
from gensim import corpora


# Actualizar y limpiar el texto nuevamente
df1['clean_Abstract'] = df1['Abstract'].apply(clean_text)

# Asegúrate de que df1 tenga las columnas necesarias ('clean_Abstract' y 'tokenized_Abstract')
df1['tokenized_Abstract'] = df1['clean_Abstract'].apply(lambda x: x.split())

# Crear el diccionario y el corpus
dictionary = corpora.Dictionary(df1['tokenized_Abstract'])
corpus = [dictionary.doc2bow(text) for text in df1['tokenized_Abstract']]

# Ejecutar el modelo LDA
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Crear la visualización de pyLDAvis
vis = gensimvis.prepare(lda_model, corpus, dictionary)



# Mostrar la visualización en un expander
with st.expander("LDA Topic Modeling Visualization", expanded=True):
    # Título dentro del expander
    st.subheader("Interactive LDA Visualization")
    
    # Renderizar la visualización interactiva
    st.components.v1.html(pyLDAvis.prepared_data_to_html(vis), height=800, width=1000)



################################
    

# Crear un expander para los WordClouds
with st.expander("WordClouds for LDA Topics", expanded=True):
    

    # Crear columnas para mostrar WordClouds
    col1, col2 = st.columns(2)

    # Iterar sobre los temas para generar los WordClouds
    for t in range(lda_model.num_topics):
        # Crear WordCloud para el tema actual
        wordcloud = WordCloud(background_color="white").fit_words(dict(lda_model.show_topic(t, 200)))
        
        # Crear la figura de Matplotlib
        plt.figure()
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.title(f"Topic {t + 1}")  # Cambiar título a "Topic 1", "Topic 2", etc.

        # Mostrar WordCloud en la columna correspondiente
        if t % 2 == 0:  # Si el índice es par, usar la primera columna
            with col1:
                st.pyplot(plt)
        else:  # Si el índice es impar, usar la segunda columna
            with col2:
                st.pyplot(plt)




###########################

from sklearn.cluster import KMeans

k_optimal=2

kmeans = KMeans(n_clusters=k_optimal)
clusters = kmeans.fit_predict(tfidf_df)
df1['Cluster'] = clusters


# Get topic distribution for each document
doc_topic_dist = [lda_model.get_document_topics(bow) for bow in corpus]

# Initialize a matrix with zeros
doc_topic_matrix = np.zeros((len(doc_topic_dist), lda_model.num_topics))

# Populate the matrix
for i, doc in enumerate(doc_topic_dist):
    for topic, prob in doc:
        doc_topic_matrix[i, topic] = prob
        


from sklearn.decomposition import PCA 

# Perform K-means clustering (let's assume 3 clusters for this example)
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(doc_topic_matrix)

# Add cluster labels back to your original DataFrame
df1['Topic_Cluster'] = clusters


# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(doc_topic_matrix)
    


# Mostrar el análisis en un expander
with st.expander("Topic Cluster Visualization", expanded=True):
    
    
    # Crear el gráfico de dispersión
    plt.figure(figsize=(8, 6))
    plt.scatter(
        reduced_features[:, 0], 
        reduced_features[:, 1], 
        c=clusters, 
        cmap='viridis', 
        s=50, 
        alpha=0.7
    )
    plt.colorbar(label='Cluster')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Topic Cluster Visualization using PCA')
    
    # Mostrar el gráfico en Streamlit
    st.pyplot(plt)    