import pandas as pd
import numpy as np
import operator
import pickle
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker
from flask import Flask, jsonify, request, render_template
from datetime import datetime, timedelta

app = Flask(__name__)

stop_words = set(stopwords.words("english"))
spell = SpellChecker()

df = pd.read_csv("C:/Users/visha/data science/IIB_Combined_Data.csv")
#df = df.dropna()
df = df.dropna(subset=["Compl_Details"])
df["Compl_Details"]=df["Compl_Details"].apply(lambda x : str(x).lower())
database = df.copy()
col = df.columns
exclude_dict = {}
vectorizer = TfidfVectorizer(stop_words="english")
tfidf = vectorizer.fit_transform(database["Compl_Details"])
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return set(synonyms)

@app.route('/')
def home():
    return render_template('front1.html')

@app.route('/search', methods=['GET','POST'])
def semantic_search():
    if request.method == 'POST':
        exclude_dict ={'health': ['wellness', 'third-party administrator', 'tpa', 'floater', 'top up', 'portability', 'mediclaim', 
               'convalescence benefit', 'waiting period', 'aarogya', 'arogya', 'ayurvedic', 'rakshak', 'health card', 
               'health checkup', 'body checkup', 'copayment', 'network hospitals', 'ambulance', 'network provider',
               'hospitalization', 'bed', 'accommodation', 'medical', 'manipal cigna', 'aditya birla', 'religare',
               'niva bupa', 'maxbupa', 'star', 'heritage', 'star & allied insurance', 'disease', 'illness', 
               'terminal illness', 'treatment',  'in-patient treatment', 'comorbidities', 'pre-existing diseases', 
               'ayush', 'sickness', 'surgery', 'fever', 'cataract', 'first diagnosis', 'home nursing',
               'domiciliary hospitalization', 'care health', 'emergency care', 'medical practitioner', 'doctor', 
               'prescription', 'child delivery', 'consultation fee', 'icu charge', 'intensive care unit',
               'waiting period', 'room rent', 'out-patient department', 'opd', 'sub-limit', 
               'pre hospitalization', 'post hospitalization', 'reimbursement claims'],
                'claim': ['claim'],
                'life': ['life'],
                'mis_selling': ['mis selling','fraud sells', 'agent cheat', 'fake policy sell'],
                'misselling': ['mis selling','fraud sells', 'agent cheat', 'fake policy sell'],
                'mis selling': ['mis selling','fraud sells', 'agent cheat', 'fake policy sell'],
                'missell': ['mis selling','fraud sells', 'agent cheat', 'fake policy sell'],
                'fraud_sells': ['fraud sells','mis selling', 'agent cheat', 'fake policy sell'],
                'agent_cheat': ['agent cheat','mis selling', 'fraud sells', 'fake policy sell'],
                'fake_policy_sell': ['fake policy sell','mis selling', 'fraud sells', 'agent cheat']}
        # Perform search and get results
        # query = request.json['query']
        
        query = request.form['query']
        query = query.lower()
        time = request.form['time']
        
        # Split the query into separate keywords and queries
        query_split = query.split('+')
        queries = []
        for q in query_split:
            q = q.strip().replace(' ', '_')
            queries.append(q)

        # Generate a set of keywords for each query based on the exclude_dict and query
        keyword_sets = []
        for q in queries:
            word_sets = {}
            for word in q.split():
                if word in exclude_dict.keys():

                    word_sets[word] = set(exclude_dict[word])
                else:
                    exclude_dict[word]= list(get_synonyms(word))
                    word_sets[word] = set(get_synonyms(word))
                    
                    if not list(get_synonyms(word)):
                        # Append the word directly to the exclude_dict with key and value as the same word
                        exclude_dict[word] = [word]
                        word_sets[word] = set([word])
                    
            all_keywords = []
            for key in word_sets.keys():
                for keyword in word_sets[key]:
                    all_keywords.append(keyword)
            keyword_sets.append(set(all_keywords))

        # Perform each query and store the results
        results_sets = []
        for ks in keyword_sets:
            query_str = " ".join(list(ks))
            query_vector = vectorizer.transform([query_str])
            similarity = cosine_similarity(tfidf, query_vector)
            results = database.copy()
            results["similarity"] = similarity
            results["Date"] = pd.to_datetime(results["compl_reg_dt"])
            
            reference_date = datetime.now().date()

            if time=='all':
                time = (reference_date - results["Date"].dt.date).dt.days[0]
            results['days']=(reference_date - results["Date"].dt.date).dt.days
            results=results.astype({'days': 'int'})
            #results['days']=results['days'].apply(lambda x : int(x))
            time = np.int32(time)
            # print(time)
            results = results[(reference_date - results["Date"].dt.date).dt.days <= time ]
            
            results = results[results["similarity"]>0.0]
            results["query_words"] = results["Compl_Details"].apply(lambda x: any(word in x for word in ks))
            results = results[results["query_words"]]
            results_sets.append(set(results.index))

        # Find the intersection of all the result sets to get the common results
        #common_results = set.union(*results_sets)
        common_results = set.intersection(*results_sets)
        results = database.loc[common_results]
        num_rows = len(results)

        return render_template('front1.html', tables=[results[['IRDA_Token_num', 'Compl_Details']].to_html(classes='data')], titles=results[['IRDA_Token_num', 'Compl_Details']], num_rows=num_rows, query=query)
        
    else:
        # Render template without any data
        return render_template('front1.html', tables=[], titles=[])
    
if __name__ == '__main__':
    
    app.run(debug=True)
