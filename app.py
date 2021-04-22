from flask import Flask, request,render_template,redirect
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

vectorizer = pickle.load(open('vectorizer.pickle','rb'))
data = pd.read_csv('book_data_rating_pages.csv')
true_data = pd.read_csv('book_data.csv')

ncorpus = vectorizer.transform(data['clean_data'].tolist())
similarity = cosine_similarity(ncorpus)
names = data['book_title'].tolist()

def get_recom_cosin(title,func=similarity):
  idx = names.index(title)
  similar = func[idx]
  values= []
  for n,i in enumerate(similar):
    values.append((n,i))
  #values= list(enumerate(similar))
  scores = sorted(values,key = lambda x : x[1],reverse=True)
  scores = scores[1:30]
  idxs = [i[0] for i in scores]
  return  true_data.iloc[idxs]

@app.route("/",methods=['POST','Get'])
def home():
	if request.method == "POST":
		title = request.form['title'].strip()
		df = get_recom_cosin(title)
		titles = df['book_title'].tolist()
		pages = df['book_pages'].tolist()
		desp = df['book_desc'].tolist()
		author = df['book_authors'].tolist()
		no = len(titles)
		return render_template('home.html',ans = [no,titles,pages,desp,author])
	else:
		return render_template('home.html',ans = "no")


if __name__ == '__main__':
	app.run(debug=False)