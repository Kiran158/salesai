from openai import OpenAI
from app.config import settings
 
## Connect to OpenAI client
client = OpenAI(api_key = settings.OPENAI_API_KEY)
 
## Creating single embedding vector for given text (single)
## args text: input text to embed
## returns: List[float] representing embedded vector
def embed_text(text:str) -> list[float]:
    ## Avoid complete empty strings and hence replace with space char/text
    if not text or text.strip() == "":
        text = " "
   
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text    
    )
   
    return response.data[0].embedding
 
## Creating embedding vectors for given batch of texts
## args text: input list of texts
## returns: embedding vectors
def embed_texts(texts:list[str]) -> list[list[float]]:
    if not texts:
        return []
   
    ## Avoid any empty text elements and hence replace with space char/text
    cleaned = [t if t and t.strip() != "" else " " for t in texts]
 
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=cleaned
    )
 
    vectors: list[list[float]] = []
 
    for item in response.data:
        vectors.append(item.embedding)
 
    return vectors