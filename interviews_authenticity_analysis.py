import nltk
import spacy
import pandas as pd
import re
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
from gensim import corpora
from gensim.models import LdaModel

# Laden der spaCy-Modell für die deutsche Sprache
nlp = spacy.load('de_core_news_sm')

# 1. Textbereinigung
def clean_text(text):
    # Umwandlung in Kleinbuchstaben
    text = text.lower()
    # Entfernen von Satzzeichen und Zahlen
    text = re.sub(r'[^\w\s]', '', text)
    # Entfernen von Ziffern
    text = re.sub(r'\d+', '', text)
    return text

# 2. Lemmatisierung und Entfernen von Stoppwörtern
def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

# 3. Verarbeitung der Datei
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Bereinigung und Vorverarbeitung des Textes
    cleaned_text = clean_text(text)
    preprocessed_text = preprocess_text(cleaned_text)
    
    return preprocessed_text

# 4. Erstellung der Wortwolke
def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# 5. Netzwerk-Analyse (Kookkurrenz der Wörter)
def create_network(text, threshold=500):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    words = vectorizer.get_feature_names_out()
    
    # Erstellen des Graphen
    G = nx.Graph()
    
    # Kookkurrenz von Wörtern hinzufügen, wobei Verbindungen mit geringem Vorkommen ausgeschlossen werden
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i != j:
                count = X[0, vectorizer.vocabulary_.get(word1)] * X[0, vectorizer.vocabulary_.get(word2)]
                if count >= threshold:  # Schwellenwert für die Häufigkeit der Verbindungen
                    G.add_edge(word1, word2, weight=count)
    
    # Zeichnen des Graphen
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=10, font_color='black', edge_color=weights, edge_cmap=plt.cm.Blues)
    plt.show()

# 6. Themenmodellierung (LDA)
def topic_modeling(text, num_topics=5):
    # Erstellung des Korpus
    texts = [text.split()]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # LDA-Modell
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    
    # Ausgeben der Themen
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)

# Hauptfunktion
def main():
    file_path = '/Users/rsmirnov/Desktop/MApp_November/probanden_clean.txt'  # Pfad zur Datei
    
    text = process_file(file_path)
    
    # Erstellung der Wortwolke
    create_wordcloud(text)
    
    # Netzwerk-Analyse
    create_network(text)
    
    # Themenmodellierung
    topic_modeling(text)

# Ausführung der Hauptfunktion
main()
