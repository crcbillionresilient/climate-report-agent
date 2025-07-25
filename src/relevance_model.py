"""
Weekly training script (run by train.yml):
1. Reads data/reports_master.json + any CSVs in data/labels/
2. Uses MiniLM embeddings + logistic regression
3. Uploads model to GitHub Release (handled by workflow)
"""
import os, json, glob, joblib, pathlib, datetime, hashlib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA = pathlib.Path("data")
LABELS = DATA / "labels"
MODEL_OUT = DATA / "model_latest.joblib"
model = SentenceTransformer(os.getenv("EMBEDDING_MODEL_NAME","all-MiniLM-L6-v2"))

def load_embeddings(recs):
    vecs, y = [], []
    for r in recs:
        vecs.append(model.encode(r["summary"]))
        y.append(1 if r["label"]=="positive" else 0)
    return vecs, y

def main():
    if not LABELS.exists(): print("No labels yet."); return
    master=json.load(open(DATA/"reports_master.json"))
    labelmap={}
    for csv in glob.glob(str(LABELS/"*.csv")):
        for line in open(csv):
            sha,lbl=line.strip().split(",")
            labelmap[sha]=lbl
    labeled=[r for r in master if r["sha"] in labelmap]
    for r in labeled: r["label"]=labelmap[r["sha"]]
    if len(labeled)<10: print("Need ≥10 labeled docs."); return
    X,y=load_embeddings(labeled)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    clf=LogisticRegression(max_iter=1000).fit(Xtr,ytr)
    print(classification_report(yte,clf.predict(Xte)))
    joblib.dump(clf, MODEL_OUT)
    print("Model saved →", MODEL_OUT)

if __name__=="__main__": main()
