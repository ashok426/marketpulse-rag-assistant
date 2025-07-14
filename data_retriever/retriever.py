import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

class Retriever:
    def __init__(self, qdrant_host, qdrant_port, collection_name, embedding_model_name="text-embedding-3-small"):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model=embedding_model_name, dimensions=512)
        self.llm = ChatOpenAI(model="gpt-4o")

    def get_top_k_chunks(self, query, top_k=20):
        query_embedding = self.embeddings.embed_query(query)
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=True
        )
        results = []
        for hit in search_result:
            results.append({
                'page_content': hit.payload['page_content'],
                'embedding': hit.vector,
                'score': hit.score
            })
        return results, np.array(query_embedding)

    def mmr(self, query_embedding, docs, lambda_mult=0.7, k=5):
        doc_embeddings = np.array([doc["embedding"] for doc in docs])
        doc_scores = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        doc_doc_sim = np.dot(doc_embeddings, doc_embeddings.T)
        doc_doc_sim /= (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) @ np.linalg.norm(doc_embeddings, axis=1, keepdims=True).T + 1e-8)
        selected, unselected = [], list(range(len(docs)))
        for _ in range(min(k, len(docs))):
            mmr_scores = []
            for idx in unselected:
                relevance = doc_scores[idx]
                diversity = max([doc_doc_sim[idx, sel_idx] for sel_idx in selected]) if selected else 0
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity
                mmr_scores.append((mmr_score, idx))
            _, selected_idx = max(mmr_scores, key=lambda x: x[0])
            selected.append(selected_idx)
            unselected.remove(selected_idx)
        return [docs[i] for i in selected]

    def generate_llm_response(self, query, context_chunks):
        PROMPT = PromptTemplate.from_template(
            """You are a helpful assistant. Answer the user's question using only the given context below.
            create a complete answer only from the given context. You should not create your own answer which is outside the context.

            Context:
            {context}

            Question:
            {question}

            Answer:"""
        )
        context = "\n\n".join(context_chunks) if context_chunks else "No context available."
        message = [
            {"role": "user", "content": PROMPT.format(context=context, question=query)}
        ]
        resp = self.llm.invoke(message)
        return resp.content.strip()

    def rag_pipeline(self, user_query, top_k=20, mmr_k=5):
        docs, query_emb = self.get_top_k_chunks(user_query, top_k=top_k)
        if not docs:
            print("No relevant chunks found.")
            return ""
        reranked_chunks = self.mmr(query_emb, docs, lambda_mult=0.7, k=mmr_k)
        context_chunks = [doc['page_content'] for doc in reranked_chunks]
        answer = self.generate_llm_response(user_query, context_chunks)
        return answer
    
if __name__ == "__main__":
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    COLLECTION_NAME = "document_chunks_rag"

    retriever = Retriever(QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME)

    print("Enter your query (or type 'exit' to quit):")
    while True:
        user_query = input("Query: ")
        if user_query.lower() == "exit":
            break
        answer = retriever.rag_pipeline(user_query, top_k=15, mmr_k=5)
        print("\n--- LLM Final Answer ---\n", answer)


