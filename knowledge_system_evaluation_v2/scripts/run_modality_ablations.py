import asyncio
import json
import logging
from unittest.mock import patch

import numpy as np
from sentence_transformers import SentenceTransformer

from knowledge_inference.service import InferenceService
from knowledge_inference.types import RetrievalHit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_chunk_text(text: str) -> tuple[str, str]:
    caption_idx = text.find("Caption:")
    transcript_idx = text.find("Transcript:")
    
    caption_text = ""
    transcript_text = ""
    
    if caption_idx != -1 and transcript_idx != -1:
        if caption_idx < transcript_idx:
            caption_text = text[caption_idx:transcript_idx].replace("Caption:", "").strip()
            transcript_text = text[transcript_idx:].replace("Transcript:", "").strip()
        else:
            transcript_text = text[transcript_idx:caption_idx].replace("Transcript:", "").strip()
            caption_text = text[caption_idx:].replace("Caption:", "").strip()
    elif caption_idx != -1:
        caption_text = text.replace("Caption:", "").strip()
    elif transcript_idx != -1:
        transcript_text = text.replace("Transcript:", "").strip()
    else:
        transcript_text = text
        
    return transcript_text, caption_text

async def process_modality_ablations():
    service = InferenceService()
    service.initialize()
    
    logger.info("Extracting modalities from chunks...")
    chunk_refs = []
    asr_texts = []
    vlm_texts = []
    
    for store_name, store in service.stores.items():
        for chunk_id, chunk in store.chunks_kv.items():
            text = str(chunk.get("content", ""))
            asr, vlm = split_chunk_text(text)
            
            chunk_refs.append((store, chunk_id, chunk.get("video_segment_id", [])))
            asr_texts.append(asr)
            vlm_texts.append(vlm)
            
    logger.info("Loading embedding model...")
    # Normalize embeddings to easily compute cosine similarity via dot product
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    
    logger.info("Embedding ASR texts...")
    asr_embeddings = model.encode(asr_texts, normalize_embeddings=True)
    
    logger.info("Embedding VLM texts...")
    vlm_embeddings = model.encode(vlm_texts, normalize_embeddings=True)
    
    def retrieve_top_k(query_emb, doc_embeddings, texts, k=15):
        scores = np.dot(doc_embeddings, query_emb)
        top_n = np.argsort(scores)[::-1][:k]
        
        hits = []
        for i in top_n:
            if not texts[i].strip():
                continue
            store, chunk_id, seg_ids = chunk_refs[i]
            if isinstance(seg_ids, str):
                seg_ids = [seg_ids]
            hits.append(RetrievalHit(
                chunk_id=chunk_id,
                video_name=store.video_name,
                source="vector_modality",
                chunk_text=texts[i],
                segment_ids=[str(x) for x in seg_ids],
                score_semantic=float(scores[i]),
                score_entity=0.0,
                score_graph=0.0,
            ))
        return hits[:k]
        
    input_file = "knowledge_system_evaluation_v2/community_qa_dataset_ablations.json"
    with open(input_file, "r") as f:
        data = json.load(f)
        
    for i, item in enumerate(data):
        logger.info(f"Processing {i+1}/{len(data)}: {item.get('question_title', 'Unknown')}")
        
        # Skip if already done
        if "asr_only" in item.get("ablations", {}) and "vision_only" in item.get("ablations", {}):
            continue
            
        query = item.get("question_title", "") + " " + item.get("question_body", "")
        query_emb = model.encode([query], normalize_embeddings=True)[0]
        
        # ASR Only
        asr_hits = retrieve_top_k(query_emb, asr_embeddings, asr_texts)
        async def mock_asr(*args, **kwargs):
            return asr_hits
        with patch("knowledge_inference.service.retrieve_all", new=mock_asr):
            asr_res = await service._answer_async(query)
            
        # Vision Only
        vlm_hits = retrieve_top_k(query_emb, vlm_embeddings, vlm_texts)
        async def mock_vlm(*args, **kwargs):
            return vlm_hits
        with patch("knowledge_inference.service.retrieve_all", new=mock_vlm):
            vlm_res = await service._answer_async(query)
            
        item["ablations"]["asr_only"] = {
            "answer": asr_res.answer,
            "evidence_sources": [e.source for e in asr_res.evidence]
        }
        item["ablations"]["vision_only"] = {
            "answer": vlm_res.answer,
            "evidence_sources": [e.source for e in vlm_res.evidence]
        }
        
        # Save checkpoints safely
        if (i + 1) % 5 == 0 or (i + 1) == len(data):
            with open(input_file, "w") as f:
                json.dump(data, f, indent=4)
                
    logger.info("Done!")

if __name__ == "__main__":
    asyncio.run(process_modality_ablations())
