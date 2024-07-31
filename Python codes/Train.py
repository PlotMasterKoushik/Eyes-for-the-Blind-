from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

print("loading  embeddings...")
data = pickle.loads(open("output/embeddings.pickle", "rb").read())