from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn import linear_model
import datetime
import pandas as pd
import pickle

"""
Word frequency analysis of text extracted for solar equipments from one line diagrams
"""

from sklearn.model_selection import train_test_split


def read_strings(file_name):
    df = pd.read_csv(file_name)
    return df


def calculate_word_vector(string_list: [str]):
    vectorizer = CountVectorizer().fit(string_list)
    # Save vectorizer.vocabulary_
    pickle.dump(vectorizer.vocabulary_, open("eq.feature.pkl", "wb"))
    return vectorizer.transform(string_list)


def calculate_hash_vector(string_list: [str]):
    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18,
                                   alternate_sign=False)
    return vectorizer.transform(string_list)


def neural_network(file_name):
    text_strings = read_strings(file_name)
    X = calculate_hash_vector(text_strings['String'])
    y = text_strings['Matching']

    clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3, verbose=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("SGDC Score = ", score)

    pickle.dump(clf, open("eq.clf.model.pkl", 'wb'))

    predictions = clf.predict(X)
    titles = ["Rejected", "Accepted"]

    with open("nn_equipment_CGDClassifier_{}".format(str(datetime.datetime.now()).replace(':', '')), 'w') as rf:

        for i in range(2):
            rf.write("-" * 30 + "\n", )
            rf.write(titles[i] + "\n")
            rf.write("-" * 30 + "\n")
            for p in range(len(predictions)):
                if predictions[p] == i:
                    rf.write(str(text_strings['String'][p]) + "\n")


def update_model(text_string: object) -> object:
    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18,
                                   alternate_sign=False)
    X_add = vectorizer.transform(text_string)
    loaded_model = pickle.load(open("eq.clf.model.pkl", 'rb'))
    loaded_model.partial_fit(X_add, [1, 1, 1, 1, 1, 1])
    pickle.dump(loaded_model, open("new.eq.clf.model.pkl", 'wb'))


def test_nn(model_file, data_file):
    # Load vocab
    text_strings = read_strings(data_file)
    text_strings['String'].str.replace('"', '')

    # Use this for the CountVectorizer that needs to be saved and loaded.
    # loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("feature.pkl", "rb")))
    # X_test = loaded_vec.fit_transform(text_strings['address'])

    # Use this for the HashingVectorizer that does not need to be saved, but... makes... MLPClassifier... slow...
    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 18,
                                   alternate_sign=False)
    X_test = vectorizer.transform(text_strings['String'])

    # load the model from disk
    loaded_model = pickle.load(open(model_file, 'rb'))

    predictions = loaded_model.predict(X_test)
    titles = ["Rejected", "Accepted"]

    with open("nn_LOAD_equipment_CGDClassifier_{}".format(str(datetime.datetime.now()).replace(':', '')), 'w') as rf:

        for i in range(2):
            rf.write("-" * 30 + "\n", )
            rf.write(titles[i] + "\n")
            rf.write("-" * 30 + "\n")
            for p in range(len(predictions)):
                if predictions[p] == i:
                    rf.write(str(text_strings['String'][p]) + "\n")


if __name__ == "__main__":
    # neural_network("equipment_training_set.csv")
    # test_nn("eq.clf.model.pkl", "equipment_testing_set.csv")
    # create_data_file("auto-extracted-results.json")
    update_model(['( 26 ) MISSION SOLAR MSE300SQ5T 300-W MODULES TOTAL',
                  '( 24 ) MISSION SOLAR MSE300SQ5T Module',
                  '24-Mission 300w panels MSE300SQ5T',
                  '300w panels MSE300SQ5T 24-Optimizers P320-54C4ARS 01-Solaredge',
                  '24-Mission Solar MSE300SQ5T',
                  'Solar MSE300SQ5T'])
