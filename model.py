import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,LlamaTokenizer, LlamaForCausalLM
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import pipeline

class ClaimClassifier:
    def __init__(self, file_path, model_name='openlm-research/open_llama_13b'):
        self.file_path = file_path
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.le = LabelEncoder()
        self.model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto',)
    
    def load_and_preprocess_data(self):
        claims = pd.read_csv(self.file_path)
        claims_texts = claims['text'].tolist()
        claims_labels = claims['category'].tolist()
        
        inputs = self.tokenizer(claims_texts, truncation=True, padding=True, max_length=512)
        labels = self.le.fit_transform(claims_labels)
        
        self.dataset = Dataset.from_dict({'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': labels})
        self.dataset = self.dataset.train_test_split(test_size=0.1)
    
    def train_model(self):
        training_args = TrainingArguments(output_dir='output', num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=64)
        self.trainer = Trainer(model=self.model, args=training_args, train_dataset=self.dataset['train'], eval_dataset=self.dataset['test'])
        self.trainer.train()
        metrics = self.trainer.evaluate()
        return metrics

    def classify_claim(self, new_claim):
        inputs = self.tokenizer(new_claim, return_tensors='pt')
        prediction = self.model(**inputs).logits
        category = self.le.inverse_transform(prediction.argmax(dim=-1).detach().numpy())[0]
        return category

class NamedEntityRecognizer:
    def __init__(self, model_name='dbmdz/bert-large-cased-finetuned-conll03-english'):
        self.ner = pipeline('ner', model=model_name)
    
    def extract_entities(self, text):
        return self.ner(text)

if __name__ == "__main__":
    claim_classifier = ClaimClassifier('claims.csv')
    claim_classifier.load_and_preprocess_data()
    metrics = claim_classifier.train_model()
    print("Training Metrics: ", metrics)

    new_claim = "A car accident happened on 5th Street..."
    print(claim_classifier.classify_claim(new_claim))
    
    ner = NamedEntityRecognizer()
    print(ner.extract_entities(new_claim))
