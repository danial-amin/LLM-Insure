import pandas as pd
data=pd.read_csv("/workspaces/LLM-Insure/synthetic_claims-data.csv")
print(data.head)
data["claim_text"]=data["claim_text"].str.split(':')
data["claim_text"]=data["claim_text"].apply(lambda x:x[1])
print(data.head())
data.to_csv('claims.csv', index=False)