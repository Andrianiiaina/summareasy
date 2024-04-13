import speech_recognition as sr
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from pydub import AudioSegment
import requests
from bs4 import BeautifulSoup
from rouge_score import rouge_scorer
from sklearn.cluster import KMeans

def wikipedia(link):
    #response= requests.get(url="https://en.wikipedia.org/wiki/Eulemur")
    response= requests.get(url=link)
    soup=BeautifulSoup(response.content,'html.parser')
    main_content= soup.find('div',{'class':'mw-content-container'})
    texts=[]
    for section in main_content.find_all(['h2','h3']):
        sibling = section.find_next_sibling()
        while sibling :
            if(sibling.name=='p' and sibling.get('id','') not in ['wikitable', 'references','Bibliographie']):
                texts.append(sibling.text.strip())
            sibling = sibling.find_next_sibling()
    return ' '.join(texts)    
def audio_to_text(filename):
    #convert other format to .wav 
    sound=AudioSegment.from_file(filename)
    sound.export("file.wav",format="wav")
    #Nb: On doit decouper les audios en section de 1.5 minutes et boucler dessus.!! limiter ny length de google reco.
    with sr.AudioFile("file.wav") as source:
        audio_data= r.record(source)
        text= r.recognize_google(audio_data)
    return text    
def summarize_text(text, num_sentences=1):
    # Prétraitement du texte
    stemmer=PorterStemmer()
    text = text.lower()
    sentences = sent_tokenize(text)
    clean_sentences=[]
    for sentence in sentences:
        stop_words = set(stopwords.words('english'))
        words = [stemmer.stem(word) for word in word_tokenize(sentence) if word.isalnum() and word not in stop_words]
        clean_sentences.append(' '.join(words))
    
    # Calcul de l'importance des mots avec TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(clean_sentences)
    #calculs des scores
    #la premiere et derniere phrases sont souvent plus importante=>poids++
    max_importance=-1
    scores={}
    for i, sentence in enumerate(sentences):
        tfidf_score=tfidf_matrix[i].sum()
        if(i == 0 or i== len(clean_sentences)-1):
            tfidf_score*=1.5
        if tfidf_score > max_importance:
            max_importance=tfidf_score
        scores[tfidf_score]=sentence   
    final_text=[]    
    #print(sorted(scores, reverse=True))
    #on prend les phrases avec le score le plus haut selon l'ordre du text
    final_text=[sentence for score in scores for sc,sentence in sorted(scores.items(), reverse=True)[:num_sentences] if score==sc]
    
    #si l'ordre n'as pas d'importance: final_text=[se for score in sorted(sc, reverse=True)[:num_sentences] if score==sc] 
    
    return ' '.join(final_text)
def main():
    choices=[summarize_text,wikipedia,audio_to_text]
    x=' '
    while(x not in ['1','2','3']):
        x=input("Tapez 1 pour resumé un text\nTapez 2 pour entrer un lien vers wikipedia \nTapez 3 pour resumer un fichier audio\n")
    #++conditions à faire
    if x== '1':
        texte=input("Entrer votre text: ")
           
    if x== '2':
       lien=input("Entrer le lien wikipedia: ")
       texte= wikipedia(lien)
    if x== '3':   
        filename=input("Entrer le nom du fichier: ")
        texte=audio_to_text(filename)
    print(summarize_text(texte))    


text="""
    Generative AI, a subset of artificial intelligence, encompasses a diverse range of algorithms and techniques aimed at creating content autonomously. One prominent approach is Generative Adversarial Networks (GANs), where two neural networks, the generator and the discriminator, compete against each other to produce increasingly realistic outputs. Another popular method is Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) networks, which excel at generating sequences of data such as text, music, or even video.

    Generative AI has found applications in various fields, including art, music composition, and storytelling. Artists and musicians use generative models to explore new creative possibilities, generating unique pieces of art or music that push the boundaries of human imagination. Additionally, generative AI has practical applications in content generation, such as generating personalized recommendations, writing product descriptions, or even creating entire marketing campaigns.

    However, the widespread adoption of generative AI also raises ethical concerns. There is a risk of misuse, such as generating fake news articles or deepfake videos to spread misinformation. Moreover, generative models trained on biased datasets may perpetuate existing biases in their outputs, raising questions about fairness and equity.

    Despite these challenges, the potential of generative AI to revolutionize creative expression and problem-solving cannot be ignored. As researchers continue to innovate in this field, it is crucial to address ethical considerations and ensure responsible use of generative AI technology"""

human_summary="""
    Generative AI, a subset of artificial intelligence, 
    utilizes algorithms like GANs and LSTM networks to autonomously create diverse content such as art, music, and text. 
    While it offers innovative possibilities for creative expression and practical applications like content generation,    
    concerns regarding ethics and bias accompany its widespread adoption. 
    Addressing these challenges is essential to harness the full potential of generative AI responsibly."""
summary = summarize_text(text,num_sentences=4)

print('------------resume--------------\n',summary)
scorer= rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True)
scores_i=scorer.score(text, summary)
scores_j=scorer.score(text, human_summary)
print(scores_i)
print(scores_j)

  """Clustering qui n'a pas donné de bon resultat
    k=num_sentences
    kmeans= KMeans(n_clusters=k)
    kmeans.fit(tfidf_matrix)
    final_text=[]
    for cluster_id in range(k):
        cluster_sentences=[sentences[i] for i,label in enumerate(kmeans.labels_) if label==cluster_id]
        #print(f"{cluster_id+1} : {len(cluster_sentences)}")
        center=np.argmax([np.sum(tfidf_matrix[i]) for i,label in enumerate(kmeans.labels_) if label == cluster_id])
        final_text.append(cluster_sentences[center])
        #print(cluster_sentences)
   
    """