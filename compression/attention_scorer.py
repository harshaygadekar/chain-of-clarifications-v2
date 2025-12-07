"""
Attention-Based Importance Scoring

Uses model attention weights to identify important tokens/sentences.
More accurate than keyword-based heuristics as it captures what the model
actually focuses on when processing the input.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
import re
import logging

logger = logging.getLogger(__name__)


class AttentionBasedScorer:
    """
    Scores content importance using model attention patterns.
    
    Uses the attention weights from the model's last layer to determine
    which parts of the input are most important for the task.
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: str = "cuda"
    ):
        """
        Initialize the attention-based scorer.
        
        Args:
            model: Pre-loaded transformer model
            tokenizer: Pre-loaded tokenizer
            device: Computing device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def set_model(self, model: Any, tokenizer: Any):
        """Set the model and tokenizer for scoring."""
        self.model = model
        self.tokenizer = tokenizer
    
    def score_with_attention(
        self,
        text: str,
        question: str,
        max_length: int = 512
    ) -> List[float]:
        """
        Score sentences using model attention weights.
        
        Args:
            text: Context text to score
            question: Question for context
            max_length: Maximum sequence length
            
        Returns:
            List of importance scores per sentence
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("Model not set, returning uniform scores")
            sentences = self._split_sentences(text)
            return [1.0] * len(sentences)
        
        sentences = self._split_sentences(text)
        if len(sentences) == 0:
            return []
        
        try:
            # Create combined input: question + context
            combined_input = f"Question: {question}\n\nContext: {text}"
            
            # Tokenize
            inputs = self.tokenizer(
                combined_input,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(self.device)
            
            # Get attention weights
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_attentions=True
                )
            
            # Extract attention from last layer
            # Shape: (batch, heads, seq_len, seq_len)
            last_layer_attention = outputs.attentions[-1]
            
            # Average across heads and get attention to each token
            # We want attention FROM [CLS] or first token TO all other tokens
            avg_attention = last_layer_attention.mean(dim=1)  # Average heads
            token_importance = avg_attention[0, 0, :].cpu().numpy()  # First token's attention
            
            # Map token importance back to sentences
            sentence_scores = self._aggregate_to_sentences(
                token_importance,
                inputs['input_ids'][0],
                text,
                sentences
            )
            
            return sentence_scores
            
        except Exception as e:
            logger.warning(f"Attention scoring failed: {e}, using fallback")
            return [1.0] * len(sentences)
    
    def _aggregate_to_sentences(
        self,
        token_importance: Any,
        input_ids: torch.Tensor,
        text: str,
        sentences: List[str]
    ) -> List[float]:
        """
        Aggregate token-level importance to sentence-level.
        
        Args:
            token_importance: Array of token importance scores
            input_ids: Token IDs from tokenizer
            text: Original text
            sentences: List of sentences
            
        Returns:
            List of sentence importance scores
        """
        # Decode tokens to match with sentences
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        
        sentence_scores = []
        text_lower = text.lower()
        
        for sentence in sentences:
            sent_lower = sentence.lower()
            # Find approximate position in text
            start_pos = text_lower.find(sent_lower[:min(20, len(sent_lower))])
            
            if start_pos == -1:
                # Sentence not found, use average score
                sentence_scores.append(float(token_importance.mean()))
                continue
            
            # Estimate token range for this sentence
            # This is approximate since tokenization may differ
            char_ratio = start_pos / max(len(text), 1)
            token_start = int(char_ratio * len(tokens))
            
            end_pos = start_pos + len(sentence)
            char_ratio_end = end_pos / max(len(text), 1)
            token_end = int(char_ratio_end * len(tokens))
            
            # Clamp to valid range
            token_start = max(0, min(token_start, len(token_importance) - 1))
            token_end = max(token_start + 1, min(token_end, len(token_importance)))
            
            # Average importance for this sentence's tokens
            sent_importance = float(token_importance[token_start:token_end].mean())
            sentence_scores.append(sent_importance)
        
        # Normalize to 0-1
        if sentence_scores:
            max_score = max(sentence_scores)
            min_score = min(sentence_scores)
            if max_score > min_score:
                sentence_scores = [
                    (s - min_score) / (max_score - min_score)
                    for s in sentence_scores
                ]
        
        return sentence_scores
    
    def get_token_importance(
        self,
        text: str,
        question: str
    ) -> List[Tuple[str, float]]:
        """
        Get importance score for each token (for debugging/visualization).
        
        Args:
            text: Context text
            question: Question
            
        Returns:
            List of (token, importance) tuples
        """
        if self.model is None or self.tokenizer is None:
            return []
        
        combined_input = f"Question: {question}\n\nContext: {text}"
        
        inputs = self.tokenizer(
            combined_input,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        last_layer_attention = outputs.attentions[-1]
        avg_attention = last_layer_attention.mean(dim=1)
        token_importance = avg_attention[0, 0, :].cpu().numpy()
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
        
        return list(zip(tokens, token_importance.tolist()))
    
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class HybridScorer:
    """
    Combines attention-based and keyword-based scoring.
    
    Falls back to keyword scoring if attention is unavailable,
    and can weight both methods for more robust results.
    """
    
    def __init__(
        self,
        attention_weight: float = 0.6,
        keyword_weight: float = 0.4
    ):
        """
        Initialize hybrid scorer.
        
        Args:
            attention_weight: Weight for attention scores
            keyword_weight: Weight for keyword scores
        """
        self.attention_weight = attention_weight
        self.keyword_weight = keyword_weight
        self.attention_scorer = AttentionBasedScorer()
    
    def set_model(self, model: Any, tokenizer: Any):
        """Set model for attention scoring."""
        self.attention_scorer.set_model(model, tokenizer)
    
    def score_sentences(
        self,
        text: str,
        question: str,
        keywords: Optional[List[str]] = None
    ) -> List[float]:
        """
        Score sentences using hybrid approach.
        
        Args:
            text: Context text
            question: Question
            keywords: Optional keywords for keyword scoring
            
        Returns:
            Combined importance scores
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        
        # Get attention scores
        attention_scores = self.attention_scorer.score_with_attention(
            text, question
        )
        
        # Get keyword scores
        keyword_scores = self._score_by_keywords(sentences, question, keywords)
        
        # Combine scores
        combined_scores = []
        for i in range(len(sentences)):
            att_score = attention_scores[i] if i < len(attention_scores) else 0.5
            kw_score = keyword_scores[i] if i < len(keyword_scores) else 0.5
            
            combined = (
                self.attention_weight * att_score +
                self.keyword_weight * kw_score
            )
            combined_scores.append(combined)
        
        return combined_scores
    
    def _score_by_keywords(
        self,
        sentences: List[str],
        question: str,
        keywords: Optional[List[str]] = None
    ) -> List[float]:
        """Score sentences by keyword overlap."""
        if keywords is None:
            keywords = self._extract_keywords(question)
        
        scores = []
        for sent in sentences:
            sent_lower = sent.lower()
            count = sum(1 for kw in keywords if kw.lower() in sent_lower)
            scores.append(count)
        
        # Normalize
        max_score = max(scores) if scores and max(scores) > 0 else 1
        return [s / max_score for s in scores]
    
    @staticmethod
    def _extract_keywords(question: str) -> List[str]:
        """Extract keywords from question."""
        stopwords = {
            'what', 'when', 'where', 'who', 'why', 'how',
            'is', 'are', 'was', 'were', 'the', 'a', 'an',
            'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'
        }
        words = question.lower().split()
        return [w.strip('?.,!') for w in words 
                if w.lower() not in stopwords and len(w) > 2]
    
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
