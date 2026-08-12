from bert_test import BERTopic

class TopicModel:
    def __init__(self, model_path=None):
        if model_path:
            self.model = BERTopic.load(model_path)
        else:
            self.model = BERTopic()

    def fit(self, texts):
        topics, probs = self.model.fit_transform(texts)
        return topics

    def transform(self, texts):
        topics, probs = self.model.transform(texts)
        return topics

    def save(self, path):
        self.model.save(path)

    def get_info(self):
        return self.model.get_topic_info()