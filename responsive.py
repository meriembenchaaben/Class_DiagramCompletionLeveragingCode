from flask import Flask, render_template,  request, url_for, flash, redirect
from flask import request
import numpy as np
import nltk
from gensim.models import KeyedVectors
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
#nltk.download('wordnet')
#nltk.download('omw-1.4')
class Word2VecVectorizer:
    def __init__(self, model):
        print("Loading in word vectors...")
        self.word_vectors = model
        print("Finished loading in word vectors")

    def fit(self, data):
        pass

    def transform(self, data):
        # determine the dimensionality of vectors
        v = self.word_vectors.get_vector('king')
        self.D = v.shape[0]

        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.split()
            print(tokens)
            vecs = []
            m = 0
            for word in tokens:
                try:
                    # throws KeyError if word not found
                    vec = self.word_vectors.get_vector(word)
                    vecs.append(vec)
                    m += 1
                except KeyError:
                    pass
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
        return X


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    try :
        text= text.replace(",", ' ')
    except KeyError:
        pass
    ## clean (convert to lowercase and remove punctuations and     characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                    lst_stopwords]

    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list
    text = " ".join(lst_text)
    return text


import re


def intercept(Input):


    input_ids = tokenizer(Input, return_tensors='pt')
    beam_output = TrainedModel.generate(**input_ids,
                                        max_length=300,  # or 100
                                        no_repeat_ngram_size=4,
                                        early_stopping=True
                                        )

    result = tokenizer.decode(beam_output[0], skip_special_tokens=True)
    print("generated text : ", result)
    #new = result.split(Input)[1]
    new=result
    results = []

    while True:

        try:


            i = re.search(r'\((.*?)\)', new).group(1)
            results.append(i)

            new = new.split(i)[1]

        except:
            break
    ##check no repeated association:

    assert len(results) == len(set(results)), 'not_repeated classes'
    return  results


##Text generation


tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token


# relationship Prediction:

loaded_model = pickle.load(open('C:\\Users\\merie\\OneDrive\\Documents\\GEODES\\Hiver2022\\Milestone\\RandomForest_model2.sav', 'rb'))
TrainedModel = AutoModelForCausalLM.from_pretrained("C:\\Users\\merie\\OneDrive\\Documents\\GEODES\\Hiver2022\\Milestone\\fineTunedModelConceptsPrediction\\fineTunedModelConceptsPredictionLowerCase/")




w2vModel = KeyedVectors.load_word2vec_format("C:\\Users\\merie\\OneDrive\\Documents\\GEODES\\Hiver2022\\Milestone\\vectorizer")

vectorizer = Word2VecVectorizer(w2vModel)

design  = []

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')




@app.route('/design', methods=['POST', 'GET'])
def create():
    if request.method == 'POST':
            content = request.form['Concepts']
            design.append(content)
            if not content:
                flash('start designing !')

            else:

                res= intercept(content)
                return render_template(('index.html'), selection= res , approved=design)
    else:

        content = request.args.get('Concepts')

@app.route('/selection', methods = ['POST'])
def submitselect():
    selectValue = request.form.get('selectedConcept')
    if(selectValue != " all of the suggestions are not relative"):
        print('selected value ', selectValue)
        design.append(" (" + selectValue+ ")")

    return render_template(('index.html'), approved= design)



@app.route('/clean', methods=['POST'])
def cleanDesign():
    design.clear()
    return render_template(('index.html'), approved= design)

@app.route('/generateAgain', methods=['POST'])
def generate():
    thestring=""
    for i,indx in enumerate(design):


        if i==0:
            thestring= design[i]
        else:
            thestring = thestring + ',' + design[i]

    print('final result: ', thestring)
    res= intercept(thestring)

    return render_template(('index.html'), selection= res , approved= design)


@app.route('/predictA', methods=['POST'])
def predictTypeAssociation():
    selectValue = request.form.get('selectedConceptForPred')
    print('selection: ' , selectValue)
    text=utils_preprocess_text(selectValue)
    print(' for prediction: ', text)
    res = vectorizer.transform([text])

    predictionN= loaded_model.predict(res)

    if predictionN == 1 :
        prediction= 'Inherits from'
    else:
        prediction = 'Association'
    return render_template(('index.html'), prediction=prediction , conceptsPredicted=selectValue , approved=design)


if __name__ == '__main__':
    app.run(debug=True)