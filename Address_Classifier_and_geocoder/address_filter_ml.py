from sklearn.feature_extraction.text import CountVectorizer, HashingVectorize
from sklearn import linear_model
import datetime
import pandas as pd
import pickle

"""
Word frequency analysis of text extracted for solar equipments from one line diagrams
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def read_strings(file_name):
    df = pd.read_csv(file_name)
    return df


def calculate_word_vector(string_list: [str]):
    vectorizer = CountVectorizer().fit(string_list)
    # Save vectorizer.vocabulary_
    pickle.dump(vec.vocabulary_, open("feature.pkl", "wb"))
    return vectorizer.transform(string_list)

def calculate_hash_vector(string_list: [str]):
    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18,
                                   alternate_sign=False)
    return vectorizer.transform(string_list)

def neural_network(file_name):
    text_strings = read_strings(file_name)
    X = calculate_hash_vector(text_strings['address'])
    y = text_strings['value']

    mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                        beta_2=0.999, early_stopping=False, epsilon=1e-08,
                        hidden_layer_sizes=(20, 20, 20), learning_rate='constant',
                        learning_rate_init=0.0001, max_iter=1000, momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=None,
                        shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
                        verbose=True, warm_start=False)
    clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3, verbose=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    mlp.fit(X_train, y_train)
    score = mlp.score(X_test, y_test)
    print("MLP Score = ", score)

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("SGDC Score = ", score)

    pickle.dump(mlp, open("mlp.model.pkl", 'wb'))
    pickle.dump(clf, open("clf.model.pkl", 'wb'))

    predictions = clf.predict(X)
    titles = ["Rejected", "Accepted"]

    with open("nn_address_CGDClassifier_{}".format(str(datetime.datetime.now()).replace(':', '')), 'w') as rf:

        for i in range(2):
            rf.write("-" * 30 + "\n", )
            rf.write(titles[i] + "\n")
            rf.write("-" * 30 + "\n")
            for p in range(len(predictions)):
                if predictions[p] == i:
                    rf.write(str(text_strings['address'][p]) + "\n")


def create_data_file(file_name: str):
    text_strings = read_strings(file_name)
    with open("training_set.csv", 'w') as ts:
        for i, r in enumerate(text_strings['a']['f']):
            ts.write("\"{}\",{}\n".format(r, text_strings['a']['m'][i]))


def test_nn(model_file, data_file):
    # Load vocab
    text_strings = read_strings(data_file)

    # Use this for the CountVectorizer that needs to be saved and loaded.
    # loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("feature.pkl", "rb")))
    # X_test = loaded_vec.fit_transform(text_strings['address'])

    # Use this for the HashingVectorizer that does not need to be saved, but... makes... MLPClassifier... slow...
    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18,
                                   alternate_sign=False)
    X_test = vectorizer.transform(text_strings['address'])

    Y_test = text_strings['value']

    # load the model from disk
    loaded_model = pickle.load(open(model_file, 'rb'))
    result = loaded_model.score(X_test, Y_test)

    predictions = loaded_model.predict(X_test)
    titles = ["Rejected", "Accepted"]

    with open("nn_LOAD_address_CGDClassifier_{}".format(str(datetime.datetime.now()).replace(':', '')), 'w') as rf:

        for i in range(2):
            rf.write("-" * 30 + "\n", )
            rf.write(titles[i] + "\n")
            rf.write("-" * 30 + "\n")
            for p in range(len(predictions)):
                if predictions[p] == i:
                    rf.write(str(text_strings['address'][p]) + "\n")


if __name__ == "__main__":
    # neural_network("training_set.csv")
    test_nn("clf.model.pkl", "training_set.csv")
    # create_data_file("auto-extracted-results.json")
