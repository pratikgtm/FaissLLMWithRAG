import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import json
import boto3

# Load the CSV file into a DataFrame
df = pd.read_csv('covid_faq.csv')

# Extract covid questions
covid_questions = df['questions']

# Encode covid questions using SentenceTransformer
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
vectors = encoder.encode(covid_questions)

# Initialize Faiss index
vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)

user_query = input("Enter Search Query: ")
# Perform a search in Faiss index
query_vector = encoder.encode([user_query])
faiss.normalize_L2(query_vector)

k = 5
D, I = index.search(query_vector, k)

# create context to generate answer from retrieved responses
answer_context = [df.iloc[idx]['answers'] for idx in I[0]]

aws_access_key_id = "XXXX"
aws_secret_access_key = "XXXX"

boto_session = boto3.session.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key)

client = boto_session.client(
    service_name='bedrock-runtime',
    region_name="us-east-1"
)

prompt = f"""

Human:
Below is the question from user. You have to respond to the question.
{user_query}

You have to use the below context list which is provided. 
This list has relevant responses sorted in descending order of relevance matching.
{answer_context}

Assistant:
"""

body = json.dumps({
    "prompt": prompt,
    "max_tokens_to_sample": 1000,
    "temperature": 0.75
})
response = client.invoke_model(
    body=body,
    modelId="anthropic.claude-v2",
    accept='application/json',
    contentType='application/json'
)
answer = json.loads(response.get('body').read())["completion"]
print(f"Response: {answer}")
