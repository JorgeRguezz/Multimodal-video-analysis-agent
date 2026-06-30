import asyncio
import json
import logging
from unittest.mock import patch

from rank_bm25 import BM25Okapi

from knowledge_inference.service import InferenceService
from knowledge_inference.types import RetrievalHit
from knowledge_build._llm import local_llm_config
from knowledge_inference.retrievers import retrieve_chunks_dense

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_bm25_index(stores):
    logger.info("Building BM25 Index across all stores...")
    corpus = []
    chunk_refs = []
    for store_name, store in stores.items():
        for chunk_id, chunk in store.chunks_kv.items():
            text = str(chunk.get("content", ""))
            corpus.append(text.lower().split())
            chunk_refs.append((store, chunk_id, text, chunk.get("video_segment_id", [])))
            
    bm25 = BM25Okapi(corpus)
    logger.info(f"BM25 Index built with {len(corpus)} chunks.")
    return bm25, chunk_refs

async def retrieve_bm25_mock(query=None, intent=None, stores=None, global_graph=None, bm25_index=None, chunk_refs=None, **kwargs):
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:15]
    
    hits = []
    for i in top_n:
        store, chunk_id, text, seg_ids = chunk_refs[i]
        if isinstance(seg_ids, str):
            seg_ids = [seg_ids]
        hits.append(RetrievalHit(
            chunk_id=chunk_id,
            video_name=store.video_name,
            source="bm25",
            chunk_text=text,
            segment_ids=[str(x) for x in seg_ids],
            score_semantic=scores[i],
            score_entity=0.0,
            score_graph=0.0,
        ))
    return hits

async def retrieve_vector_mock(query=None, intent=None, stores=None, global_graph=None, **kwargs):
    # Call only the dense retriever from our actual retrievers module
    return await retrieve_chunks_dense(query, stores, k=15)

async def retrieve_parametric_mock(query=None, intent=None, stores=None, global_graph=None, **kwargs):
    return []

async def run_vanilla_base(query):
    # Vanilla chatbot experience, absolutely no RAG system prompt or context,
    # but we DO provide the formatting rules so it outputs correctly instead of rambling.
    vanilla_system_prompt = """You are a helpful AI assistant.

Reasoning: High

Rules:
1. Answer the user's question directly and concisely.

<|channel|>analysis<|message|>[user request]. Provide answer.<|end|>
<|start|>assistant<|channel|>final<|message|>[your response]<|return|>
"""
    res = await local_llm_config.best_model_func(
        query, 
        system_prompt=vanilla_system_prompt,
        return_metadata=True
    )
    return str(res.get("answer", "")).strip()

async def process_dataset(test_mode=False):
    service = InferenceService()
    service.initialize()
    
    bm25_index, chunk_refs = build_bm25_index(service.stores)
    
    input_file = "knowledge_system_evaluation_v2/community_qa_dataset_answerability.json"
    output_file = "knowledge_system_evaluation_v2/community_qa_dataset_ablations.json"
    
    with open(input_file, "r") as f:
        data = json.load(f)
        
    if test_mode:
        data = data[:2]
        
    results = []
    
    for i, item in enumerate(data):
        logger.info(f"Processing {i+1}/{len(data)}: {item.get('question_title', 'Unknown')}")
        query = item.get("question_title", "") + " " + item.get("question_body", "")
        
        # 1. Vanilla Base
        logger.info("  -> Running vanilla_base")
        vanilla_ans = await run_vanilla_base(query)
        
        # 2. Parametric
        logger.info("  -> Running parametric")
        with patch("knowledge_inference.service.retrieve_all", new=retrieve_parametric_mock):
            param_res = await service._answer_async(query)
            
        # 3. BM25
        logger.info("  -> Running bm25")
        async def bm25_wrapper(query=None, intent=None, stores=None, global_graph=None, **kwargs):
            return await retrieve_bm25_mock(query=query, intent=intent, stores=stores, global_graph=global_graph, bm25_index=bm25_index, chunk_refs=chunk_refs)
        with patch("knowledge_inference.service.retrieve_all", new=bm25_wrapper):
            bm25_res = await service._answer_async(query)
            
        # 4. Vector-Only
        logger.info("  -> Running vector_only")
        with patch("knowledge_inference.service.retrieve_all", new=retrieve_vector_mock):
            vector_res = await service._answer_async(query)
            
        # 5. Graph-RAG (Full)
        logger.info("  -> Running graph_rag")
        graph_res = await service._answer_async(query)
        
        ablations_output = {
            "vanilla_base": {
                "answer": vanilla_ans,
                "evidence_sources": []
            },
            "parametric": {
                "answer": param_res.answer,
                "evidence_sources": [e.source for e in param_res.evidence]
            },
            "bm25": {
                "answer": bm25_res.answer,
                "evidence_sources": [e.source for e in bm25_res.evidence]
            },
            "vector_only": {
                "answer": vector_res.answer,
                "evidence_sources": [e.source for e in vector_res.evidence]
            },
            "graph_rag": {
                "answer": graph_res.answer,
                "evidence_sources": [e.source for e in graph_res.evidence]
            }
        }
        
        item["ablations"] = ablations_output
        results.append(item)
        
        # Save checkpoints
        if (i + 1) % 5 == 0 or (i + 1) == len(data):
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
                
    logger.info("Done!")

if __name__ == "__main__":
    import sys
    test_mode = "--test" in sys.argv
    asyncio.run(process_dataset(test_mode=test_mode))
