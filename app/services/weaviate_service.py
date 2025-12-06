import weaviate
from weaviate.classes.config import Property, DataType, Configure
 
from app.config import settings
from app.services.embedding_service import embed_text
 
def get_weaviate_client() -> weaviate.WeaviateClient:
 
    ## connect to weaviate cloud url and api_key from .env
    auth = weaviate.auth.AuthApiKey(api_key=settings.WEAVIATE_API_KEY)
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=settings.WEAVIATE_URL,
        auth_credentials=auth,
        headers={}              # Not using server side modules, hence empty dict
    )
 
    return client
 
 
def ensure_salesdeal_Schema() -> None:
 
    ## create the sale deals collection in Weaviate if it does not exist
    wv_client = get_weaviate_client()
 
    existing_names = wv_client.collections.list_all()
 
    if "SalesDeal" in existing_names:
        return
   
    wv_client.collections.create(
        name = "SalesDeal",
        properties = [
            Property(name="orderId", data_type=DataType.TEXT),
            Property(name="orderDate", data_type=DataType.TEXT),
            Property(name="region", data_type=DataType.TEXT),
            Property(name="productCategory", data_type=DataType.TEXT),
            Property(name="productName", data_type=DataType.TEXT),
            Property(name="customerSegment", data_type=DataType.TEXT),
            Property(name="salesChannel", data_type=DataType.TEXT),
            Property(name="salesRep", data_type=DataType.TEXT),
            Property(name="unitsSold", data_type=DataType.INT),
            Property(name="discountPercent", data_type=DataType.NUMBER),
            Property(name="revenueInr", data_type=DataType.NUMBER),
            Property(name="dealNotes", data_type=DataType.TEXT),
        ],
        vectorizer_config = Configure.Vectorizer.none(),     # We provide vectors for ourselves and not using weaviate-based vectoriser
    )
 
## perform semantic search over the sales deal collection using openai embeddings
## args (query: querying natural lang text to search by meaning,top_k: how many results to return)
## returns: list of dict with deal properties and distance
def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Perform semantic search over the 'SalesDeal' collection.
 
    - Uses OpenAI embeddings (text-embedding-3-small)
    - Queries Weaviate Cloud with nearVector
    - Safely handles cases where there are no results
    """
    # Make sure the collection exists
    ensure_salesdeal_Schema()
 
    client = get_weaviate_client()
    collection = client.collections.get("SalesDeal")
 
    # Get embedding for the query
    query_vector = embed_text(query)
 
    # Run the nearVector query
    result = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        return_properties=[
            "orderId",
            "productName",
            "revenueInr",
            "dealNotes",
            "region",
            "customerSegment",
        ],
        include_vector=False,
        return_metadata=["distance"],
    )
 
    hits: list[dict] = []
 
   
    if result is None:
        return hits
 
    objs = getattr(result, "objects", None)
    if not objs:
        # No objects returned (empty result set)
        return hits
 
   
    for obj in objs:
        props = getattr(obj, "properties", {}) or {}
        additional = getattr(obj, "additional", {}) or {}
 
        hits.append(
            {
                "order_id": props.get("orderId"),
                "product_name": props.get("productName"),
                "revenue_inr": props.get("revenueInr"),
                "deal_notes": props.get("dealNotes"),
                "region": props.get("region"),
                "customer_segment": props.get("customerSegment"),
                "distance": obj.metadata["distance"]
            }
        )
 
    return hits