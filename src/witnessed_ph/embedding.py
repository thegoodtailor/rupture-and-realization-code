"""
Witnessed Persistent Homology: Embedding Extraction
====================================================

This module handles the transformation from raw text to the point cloud P_τ.

Pipeline:
1. Segment text into utterances (measurement locations)
2. Tokenize with spaCy, filter by POS and length
3. Assign unique TokenIDs to each occurrence
4. Extract contextual embeddings using DeBERTa
5. Normalize to unit sphere S^{d-1}

The output P_τ = {ê_t : t ∈ TokenID(τ)} ⊂ S^{d-1} is the geometric
material for the Čech filtration.

References:
    Chapter 4, Section 4.6 (From dialogue tokens to a point cloud)
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from .schema import (
    Token, Utterance, Config, PointCloudData, default_config
)


# =============================================================================
# Text Segmentation
# =============================================================================

def segment_into_utterances(
    text: str,
    mode: str = "lines"
) -> List[Tuple[str, Optional[str], str]]:
    """
    Segment raw text into utterances (measurement locations).
    
    Parameters
    ----------
    text : str
        Raw input text.
    mode : str
        Segmentation mode:
        - "lines": Each non-empty line is an utterance
        - "turns": Parse "Speaker: text" format
        - "sentences": Use sentence boundaries (requires spaCy)
    
    Returns
    -------
    List of (utterance_id, speaker, text) tuples.
    
    Notes
    -----
    From Chapter 4: "Each non-empty line in the transcript is treated as
    an utterance. For this dialogue there are 49 such utterances."
    """
    utterances = []
    
    if mode == "lines":
        lines = text.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                utterances.append((f"u{i}", None, line))
                
    elif mode == "turns":
        # Parse "Speaker: text" or "**Speaker**: text" format
        lines = text.strip().split('\n')
        utt_idx = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Try to extract speaker
            speaker_match = re.match(r'^(?:\*\*)?([^:*]+)(?:\*\*)?:\s*(.+)$', line)
            if speaker_match:
                speaker = speaker_match.group(1).strip()
                content = speaker_match.group(2).strip()
            else:
                speaker = None
                content = line
            if content:
                utterances.append((f"u{utt_idx}", speaker, content))
                utt_idx += 1
                
    elif mode == "sentences":
        # Requires spaCy - handled in tokenize_and_embed
        raise NotImplementedError(
            "Sentence segmentation requires spaCy integration. "
            "Use 'lines' or 'turns' mode, or call tokenize_and_embed directly."
        )
    
    return utterances


# =============================================================================
# Tokenization with spaCy
# =============================================================================

def load_spacy_model(model_name: str = "en_core_web_sm"):
    """
    Load spaCy model, downloading if necessary.
    
    Returns the loaded model.
    """
    import spacy
    try:
        nlp = spacy.load(model_name)
    except OSError:
        # Model not installed, try downloading
        from spacy.cli import download
        download(model_name)
        nlp = spacy.load(model_name)
    return nlp


def tokenize_utterances(
    utterances: List[Tuple[str, Optional[str], str]],
    config: Config,
    nlp = None
) -> Tuple[Dict[str, Token], Dict[str, Utterance]]:
    """
    Tokenize utterances and filter by POS tags.
    
    Parameters
    ----------
    utterances : list
        List of (id, speaker, text) tuples from segment_into_utterances.
    config : Config
        Configuration with pos_filter, min_token_len, use_lemmas, stopwords.
    nlp : spacy.Language, optional
        Pre-loaded spaCy model. If None, loads en_core_web_sm.
    
    Returns
    -------
    tokens : Dict[str, Token]
        Mapping from token ID to Token (embedding field will be empty).
    utterances_dict : Dict[str, Utterance]
        Mapping from utterance ID to Utterance (embedding field will be empty).
    
    Notes
    -----
    From Chapter 4: "Within each utterance we extract content-bearing tokens
    (roughly: lemmatised nouns, verbs, adjectives and key multiword
    expressions). We keep token *occurrences*, not types."
    """
    if nlp is None:
        nlp = load_spacy_model()
    
    pos_filter = set(config["pos_filter"])
    min_len = config["min_token_len"]
    use_lemmas = config["use_lemmas"]
    
    # Get stopwords (case-insensitive)
    stopwords = set(w.lower() for w in config.get("stopwords", []))
    
    tokens: Dict[str, Token] = {}
    utterances_dict: Dict[str, Utterance] = {}
    
    for utt_id, speaker, text in utterances:
        doc = nlp(text)
        utt_token_ids = []
        tok_idx = 0
        
        for spacy_tok in doc:
            # Filter by POS
            if spacy_tok.pos_ not in pos_filter:
                continue
            # Filter by length
            if len(spacy_tok.text) < min_len:
                continue
            # Skip punctuation and whitespace
            if spacy_tok.is_punct or spacy_tok.is_space:
                continue
            # Skip stopwords (case-insensitive)
            if spacy_tok.text.lower() in stopwords or spacy_tok.lemma_.lower() in stopwords:
                continue
            
            # Create unique token ID
            token_id = f"{utt_id}_tok_{tok_idx}"
            tok_idx += 1
            
            # Build Token object (embedding filled later)
            token: Token = {
                "id": token_id,
                "text": spacy_tok.text,
                "lemma": spacy_tok.lemma_ if use_lemmas else spacy_tok.text,
                "pos": spacy_tok.pos_,
                "utterance_id": utt_id,
                "char_start": spacy_tok.idx,
                "char_end": spacy_tok.idx + len(spacy_tok.text),
                "embedding": np.array([])  # Placeholder
            }
            
            tokens[token_id] = token
            utt_token_ids.append(token_id)
        
        # Build Utterance object
        utt: Utterance = {
            "id": utt_id,
            "speaker": speaker,
            "text": text,
            "token_ids": utt_token_ids,
            "embedding": np.array([])  # Placeholder
        }
        utterances_dict[utt_id] = utt
    
    return tokens, utterances_dict


# =============================================================================
# Contextual Embeddings with Transformers
# =============================================================================

def load_embedding_model(model_name: str):
    """
    Load transformer model and tokenizer for embeddings.
    
    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g., "microsoft/deberta-v3-base").
    
    Returns
    -------
    model, tokenizer
    """
    from transformers import AutoModel, AutoTokenizer
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, tokenizer


def extract_token_embeddings(
    text: str,
    tokens_to_locate: List[Tuple[int, int, str]],  # (char_start, char_end, token_id)
    model,
    tokenizer,
    max_length: int = 512
) -> Dict[str, NDArray]:
    """
    Extract contextual embeddings for specific token spans.
    
    Parameters
    ----------
    text : str
        The full text to encode.
    tokens_to_locate : list
        List of (char_start, char_end, token_id) for tokens we want embeddings for.
    model : transformers model
        The transformer model.
    tokenizer : transformers tokenizer
        The tokenizer.
    max_length : int
        Maximum sequence length for the model.
    
    Returns
    -------
    Dict mapping token_id to embedding array.
    
    Notes
    -----
    We use the transformer's subword tokenization and align back to our
    spaCy tokens using character offsets. For each spaCy token, we take
    the mean of its constituent subword embeddings.
    """
    import torch
    
    device = next(model.parameters()).device
    
    # Tokenize with offset mapping
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        padding=True
    )
    
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    # Get hidden states
    with torch.no_grad():
        outputs = model(**encoding)
        # Use last hidden state
        hidden_states = outputs.last_hidden_state[0].cpu().numpy()  # (seq_len, hidden_dim)
    
    # Map each spaCy token to its subword indices
    embeddings = {}
    for char_start, char_end, token_id in tokens_to_locate:
        subword_indices = []
        for idx, (start, end) in enumerate(offset_mapping):
            # Skip special tokens (start==end==0)
            if start == end == 0:
                continue
            # Check if this subword overlaps with our token span
            if start < char_end and end > char_start:
                subword_indices.append(idx)
        
        if subword_indices:
            # Mean pool the subword embeddings
            token_embedding = hidden_states[subword_indices].mean(axis=0)
        else:
            # Fallback: use [CLS] token or zeros
            token_embedding = hidden_states[0]
        
        embeddings[token_id] = token_embedding
    
    return embeddings


def compute_utterance_embedding(
    utterance: Utterance,
    tokens: Dict[str, Token]
) -> NDArray:
    """
    Compute utterance embedding as mean of its token embeddings.
    
    Parameters
    ----------
    utterance : Utterance
        The utterance object.
    tokens : Dict[str, Token]
        Token dictionary with embeddings already filled.
    
    Returns
    -------
    Pooled embedding for the utterance.
    """
    token_embeddings = []
    for tid in utterance["token_ids"]:
        if tid in tokens and tokens[tid]["embedding"].size > 0:
            token_embeddings.append(tokens[tid]["embedding"])
    
    if token_embeddings:
        return np.mean(token_embeddings, axis=0)
    else:
        # Return zero vector if no tokens
        # Get dimension from any available token
        dim = 768  # Default for DeBERTa-base
        for t in tokens.values():
            if t["embedding"].size > 0:
                dim = t["embedding"].shape[0]
                break
        return np.zeros(dim)


def normalize_to_unit_sphere(embedding: NDArray) -> NDArray:
    """
    Normalize embedding to unit norm (project onto S^{d-1}).
    
    From Chapter 4: "We normalise these to lie on the unit sphere."
    """
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding


# =============================================================================
# Main Embedding Pipeline
# =============================================================================

def text_to_point_cloud(
    text: str,
    config: Optional[Config] = None,
    segmentation_mode: str = "lines",
    nlp = None,
    embedding_model = None,
    embedding_tokenizer = None
) -> PointCloudData:
    """
    Transform raw text into the normalized point cloud P_τ.
    
    This is the main entry point for Step 0 of the pipeline.
    
    Parameters
    ----------
    text : str
        Raw input text (conversation, chapter, document).
    config : Config, optional
        Configuration. Uses defaults if not provided.
    segmentation_mode : str
        How to segment text: "lines", "turns", or "sentences".
    nlp : spacy.Language, optional
        Pre-loaded spaCy model.
    embedding_model : optional
        Pre-loaded transformer model.
    embedding_tokenizer : optional
        Pre-loaded transformer tokenizer.
    
    Returns
    -------
    PointCloudData containing:
        - embeddings: (N, d) array of unit-normalized embeddings
        - token_ids: list mapping index to token ID
        - tokens: full Token objects by ID
        - utterances: full Utterance objects by ID
    
    Notes
    -----
    From Chapter 4, Section 4.6:
    "At this point we have a point cloud of a few hundred token occurrences
    in embedding space, each linked back both to its surface form and to
    the utterance that contains it."
    """
    if config is None:
        config = default_config()
    
    # Step 1: Segment into utterances
    utterance_tuples = segment_into_utterances(text, mode=segmentation_mode)
    
    if not utterance_tuples:
        raise ValueError("No utterances found in text")
    
    # Step 2: Tokenize with spaCy
    if nlp is None:
        nlp = load_spacy_model()
    
    tokens, utterances = tokenize_utterances(utterance_tuples, config, nlp)
    
    if not tokens:
        raise ValueError(
            f"No tokens extracted. Check POS filter {config['pos_filter']} "
            f"and min_token_len {config['min_token_len']}"
        )
    
    # Step 3: Load embedding model
    if embedding_model is None or embedding_tokenizer is None:
        embedding_model, embedding_tokenizer = load_embedding_model(
            config["embedding_model"]
        )
    
    # Step 4: Extract embeddings for each utterance's tokens
    # We process utterance by utterance to maintain context
    for utt_id, utt in utterances.items():
        if not utt["token_ids"]:
            continue
        
        # Collect token spans for this utterance
        tokens_to_locate = []
        for tid in utt["token_ids"]:
            tok = tokens[tid]
            tokens_to_locate.append((tok["char_start"], tok["char_end"], tid))
        
        # Extract embeddings
        utt_embeddings = extract_token_embeddings(
            utt["text"],
            tokens_to_locate,
            embedding_model,
            embedding_tokenizer
        )
        
        # Update token objects with embeddings
        for tid, emb in utt_embeddings.items():
            tokens[tid]["embedding"] = normalize_to_unit_sphere(emb)
    
    # Step 5: Compute utterance embeddings
    for utt_id, utt in utterances.items():
        utt_emb = compute_utterance_embedding(utt, tokens)
        utt["embedding"] = normalize_to_unit_sphere(utt_emb)
    
    # Step 6: Build the point cloud array P_τ
    # Create ordered list of token IDs and corresponding embeddings
    token_ids_list = list(tokens.keys())
    embeddings_list = []
    
    for tid in token_ids_list:
        emb = tokens[tid]["embedding"]
        if emb.size == 0:
            # Should not happen, but handle gracefully
            emb = np.zeros(768)  # Default dimension
        embeddings_list.append(emb)
    
    embeddings_array = np.array(embeddings_list)  # Shape: (N, d)
    
    return {
        "embeddings": embeddings_array,
        "token_ids": token_ids_list,
        "tokens": tokens,
        "utterances": utterances
    }


# =============================================================================
# Utility Functions
# =============================================================================

def compute_pairwise_distances(
    embeddings: NDArray,
    metric: str = "cosine"
) -> NDArray:
    """
    Compute pairwise distance matrix for the point cloud.
    
    Parameters
    ----------
    embeddings : NDArray
        Shape (N, d), assumed already normalized if using cosine.
    metric : str
        "cosine" (converted to angular distance) or "euclidean".
    
    Returns
    -------
    NDArray of shape (N, N) with pairwise distances.
    
    Notes
    -----
    For cosine distance on normalized vectors:
        d(x, y) = 1 - x·y
    
    For angular distance (more geometrically meaningful on sphere):
        θ(x, y) = arccos(x·y) / π  (normalized to [0, 1])
    """
    N = embeddings.shape[0]
    
    if metric == "cosine":
        # Cosine similarity matrix (since embeddings are normalized, this is dot product)
        similarities = embeddings @ embeddings.T
        # Clip to [-1, 1] to avoid numerical issues with arccos
        similarities = np.clip(similarities, -1.0, 1.0)
        # Convert to distance: 1 - similarity
        distances = 1.0 - similarities
        # Ensure diagonal is exactly 0
        np.fill_diagonal(distances, 0.0)
    
    elif metric == "angular":
        # Angular distance: arccos(similarity) / π
        similarities = embeddings @ embeddings.T
        similarities = np.clip(similarities, -1.0, 1.0)
        distances = np.arccos(similarities) / np.pi
        np.fill_diagonal(distances, 0.0)
    
    elif metric == "euclidean":
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(embeddings, metric="euclidean"))
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances


def get_embedding_dimension(point_cloud: PointCloudData) -> int:
    """Get the embedding dimension d from the point cloud."""
    return point_cloud["embeddings"].shape[1]


def get_point_cloud_size(point_cloud: PointCloudData) -> int:
    """Get the number of points N in the point cloud."""
    return point_cloud["embeddings"].shape[0]
