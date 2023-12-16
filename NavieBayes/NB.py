import numpy as np
import warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)


class NaiveBayesClassifier:

    @staticmethod
    def fit(X, y):
        classes = np.unique(y)
        class_probs = {}
        word_probs = {}
        total_documents = len(y)

        y=np.array(y)

        for c in classes:
            class_probs[c] = np.sum(y == c) / total_documents

        for c in classes:
            class_docs = X[y == c]

            word_probs[c] = (np.sum(class_docs, axis=0) + 1) / (np.sum(class_docs) + len(class_docs[0]))

        return class_probs, word_probs

        

    @staticmethod
    def predict(X, class_probs, word_probs, classes):
        predictions = []
        
        for doc in X:
            max_chance = -float("inf")
            predicted_class = None
            for c in classes:
                chance = np.log(class_probs[c])

                for word, present in enumerate(doc):
                    if present:
                        chance += np.log(word_probs[c][word])

                if chance > max_chance:
                    max_chance = chance
                    predicted_class = c

            predictions.append(predicted_class)
        
        return predictions