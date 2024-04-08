import speech_recognition as sr
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from pydub import AudioSegment

#A remplacer par le vrai fichier
filename="file.mp3"
#convert other format to .wav 
#sound=AudioSegment.from_file(filename,format="filename.format")
#file=sound.export("file.wav",format="wav")
#sound.export(filename)

def audio_to_text(filename):
    with sr.AudioFile(filename) as source:
        audio_data= r.record(source)
        text= r.recognize_google(audio_data)
    return text    

def summarize_text(text, num_sentences=2):
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
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(clean_sentences)

    # Calcul de la somme des scores TF-IDF par phrase
    scores = tfidf_matrix.sum(axis=1)
    #on prend les phrases avec le score le plus haut selon l'ordre du text
    for tf_idf_text in sorted(zip(scores, sentences)):
        final_text=[tf_idf_text[1] for score in sorted(scores, reverse=True)[:num_sentences] if score==tf_idf_text[0]]    
    return ' '.join(final_text)

text="""The best way to evaluate the performance of a language model is to embed it in
    an application and measure how much the application improves. Such end-to-end
    evaluation is called extrinsic evaluation. Extrinsic evaluation is the only way to extrinsic
    evaluation
    know if a particular improvement in the language model (or any component) is really
    going to help the task at hand. Thus for evaluating n-gram language models that are
    a component of some task like speech recognition or machine translation, we can
    compare the performance of two candidate language models by running the speech
    recognizer or machine translator twice, once with each language model, and seeing
    which gives the more accurate transcription.
    Unfortunately, running big NLP systems end-to-end is often very expensive. Instead, it’s helpful to have a metric that can be used to quickly evaluate potential
    improvements in a language model. An intrinsic evaluation metric is one that mea- intrinsic
    evaluation
    sures the quality of a model independent of any application. In the next section we’ll
    introduce perplexity, which is the standard intrinsic metric for measuring language
    model performance, both for simple n-gram language models and for the more sophisticated neural large language models of Chapter 10.
    In order to evaluate any machine learning model, we need to have at least three
    training set distinct data sets: the training set, the development set, and the test set.
    development
    set
    test set
    The training set is the data we use to learn the parameters of our model; for
    simple n-gram language models it’s the corpus from which we get the counts that
    we normalize into the probabilities of the n-gram language model.
    The test set is a different, held-out set of data, not overlapping with the training
    set, that we use to evaluate the model. We need a separate test set to give us an
    unbiased estimate of how well the model we trained can generalize when we apply
    it to some new unknown dataset. A machine learning model that perfectly captured
    the training data, but performed terribly on any other data, wouldn’t be much use
    when it comes time to apply it to any new data or problem! We thus measure the
    quality of an n-gram model by its performance on this unseen test set or test corpus.
    How should we choose a training and test set? The test set should reflect the
    language we want to use the model for. If we’re going to use our language model
    for speech recognition of chemistry lectures, the test set should be text of chemistry
    lectures. If we’re going to use it as part of a system for translating hotel booking requests from Chinese to English, the test set should be text of hotel booking requests.
    If we want our language model to be general purpose, then the test test should be
    drawn from a wide variety of texts. In such cases we might collect a lot of texts
    from different sources, and then divide it up into a training set and a test set. It’s
    important to do the dividing carefully; if we’re building a general purpose model,
     EVALUATING LANGUAGE MODELS: PERPLEXITY
    we don’t want the test set to consist of only text from one document, or one author,
    since that wouldn’t be a good measure of general performance.
    Thus if we are given a corpus of text and want to compare the performance of
    two different n-gram models, we divide the data into training and test sets, and train
    the parameters of both models on the training set. We can then compare how well
    the two trained models fit the test set.
    But what does it mean to “fit the test set”? The standard answer is simple:
    whichever language model assigns a higher probability to the test set—which
    means it more accurately predicts the test set—is a better model. Given two probabilistic models, the better model is the one that has a tighter fit to the test data or that
    better predicts the details of the test data, and hence will assign a higher probability
    to the test data.
    Since our evaluation metric is based on test set probability, it’s important not to
    let the test sentences into the training set. Suppose we are trying to compute the
    probability of a particular “test” sentence. If our test sentence is part of the training
    corpus, we will mistakenly assign it an artificially high probability when it occurs
    in the test set. We call this situation training on the test set. Training on the test
    set introduces a bias that makes the probabilities all look too high, and causes huge
    inaccuracies in perplexity, the probability-based metric we introduce below.
    Even if we don’t train on the test set, if we test our language model on it many
    times after making different changes, we might implicitly tune to its characteristics,
    by noticing which changes seem to make the model better. For this reason, we only
    want to run our model on the test set once, or a very few number of times, once we
    are sure our model is ready.
    For this reason we normally instead have a third dataset called a development development
    test
    test set or, devset. We do all our testing on this dataset until the very end, and then
    we test on the test once to see how good our model is.
    How do we divide our data into training, development, and test sets? We want
    our test set to be as large as possible, since a small test set may be accidentally unrepresentative, but we also want as much training data as possible. At the minimum,
    we would want to pick the smallest test set that gives us enough statistical power
    to measure a statistically significant difference between two potential models. It’s
    important that the dev set be drawn from the same kind of text as the test set, since
    its goal is to measure how we would do on the test set."""
summary = summarize_text(text,num_sentences=4)
print(summary)




"""
    Autre methode plus facile mais ne prend pas en compte l'ordre des phrases 
    #print(sorted(zip(scores, sentences), reverse=True))
    # Trier les phrases par score décroissant
    ranked_sentences = [sentence for _, sentence in sorted(zip(scores, sentences), reverse=True)]
    
    # Sélectionner les premières phrases pour former le résumé
    summary = ' '.join(ranked_sentences[:num_sentences])

    return summary
    """