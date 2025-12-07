"""
Agent Chain Orchestrator

Coordinates the multi-agent pipeline: Retriever → Reasoner → Verifier
Handles context passing, compression, and result aggregation.
"""

from typing import Dict, Optional, Any, Tuple
from agents.retriever import RetrieverAgent
from agents.reasoner import ReasonerAgent
from agents.verifier import VerifierAgent
from compression.naive_compression import NaiveCompressor
from compression.role_specific import Clarifier
import time
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


class AgentChain:
    """
    Orchestrates the three-agent chain with optional compression.

    Supports multiple compression strategies:
    - No compression
    - Fixed-ratio compression
    - Role-specific adaptive compression
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        device: Optional[str] = None,
        compression_type: str = "none",
        compression_ratio: float = 0.5,
        dynamic_compression: bool = False,
        confidence_threshold: float = 0.9,
        skip_verification_if_confident: bool = False
    ):
        """
        Initialize the agent chain.

        Args:
            model_name: Model to use for all agents
            device: Computing device
            compression_type: Type of compression (none, fixed, role_specific, semantic, dynamic)
            compression_ratio: Compression ratio if applicable
            dynamic_compression: Whether to dynamically adjust compression ratio
            confidence_threshold: Threshold for skipping verification
            skip_verification_if_confident: Whether to skip verifier if reasoner is confident
        """
        self.model_name = model_name
        self.device = device
        self.compression_type = compression_type
        self.compression_ratio = compression_ratio
        self.dynamic_compression = dynamic_compression
        self.confidence_threshold = confidence_threshold
        self.skip_verification_if_confident = skip_verification_if_confident

        # Load shared model once for all agents
        logger.info("Initializing agent chain with shared model architecture...")
        self.shared_model, self.shared_tokenizer = self._load_shared_model()

        # Initialize agents with shared model
        logger.info("Initializing agents with shared model...")
        self.retriever = RetrieverAgent(
            model_name=model_name,
            device=device,
            model=self.shared_model,
            tokenizer=self.shared_tokenizer
        )
        self.reasoner = ReasonerAgent(
            model_name=model_name,
            device=device,
            model=self.shared_model,
            tokenizer=self.shared_tokenizer
        )
        self.verifier = VerifierAgent(
            model_name=model_name,
            device=device,
            model=self.shared_model,
            tokenizer=self.shared_tokenizer
        )

        # Initialize compression modules
        self._init_compression()

        logger.info(
            f"Agent chain initialized with {compression_type} compression and shared model"
        )

    def _load_shared_model(self) -> Tuple[Any, Any]:
        """
        Load the model and tokenizer once to be shared across all agents.

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading shared model: {self.model_name}")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Auto-detect device
            if self.device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            logger.info(f"Shared model will use device: {device}")

            # Detect model type
            is_t5_family = any(x in self.model_name.lower() for x in ['t5', 'flan'])

            # Determine dtype
            dtype = torch.float16 if device == "cuda" else torch.float32

            # Load model based on type
            if is_t5_family:
                logger.info(f"Loading Seq2Seq model (T5 family)")
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype
                )
            else:
                logger.info(f"Loading CausalLM model")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype
                )

            # Move to device
            if device == "cuda":
                cuda_device = torch.device("cuda:0")
                logger.info(f"Moving shared model to {cuda_device}")
                model.to(cuda_device)

                # Log GPU memory
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1024**2
                    reserved = torch.cuda.memory_reserved(0) / 1024**2
                    logger.info(f"GPU Memory after shared model load:")
                    logger.info(f"  Allocated: {allocated:.2f} MB")
                    logger.info(f"  Reserved: {reserved:.2f} MB")
            else:
                model.to(device)
                logger.info(f"Shared model loaded on CPU")

            model.eval()

            # Set pad token if needed
            if tokenizer.pad_token is None:
                if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = tokenizer.unk_token

            logger.info(f"Shared model loaded successfully")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load shared model: {e}")
            raise

    def _init_compression(self):
        """Initialize compression modules based on type."""
        if self.compression_type == "fixed":
            self.compressor_1_2 = NaiveCompressor(
                compression_ratio=self.compression_ratio,
                strategy="first_n"
            )
            self.compressor_2_3 = NaiveCompressor(
                compression_ratio=self.compression_ratio,
                strategy="first_n"
            )
        elif self.compression_type == "role_specific":
            self.clarifier_1_2 = Clarifier("retriever", "reasoner")
            self.clarifier_2_3 = Clarifier("reasoner", "verifier")
        elif self.compression_type == "dynamic":
            # Dynamic compression with adaptive ratio
            from compression.dynamic_compression import AdaptiveClarifier
            self.clarifier_1_2 = AdaptiveClarifier("retriever", "reasoner", self.compression_ratio)
            self.clarifier_2_3 = AdaptiveClarifier("reasoner", "verifier", self.compression_ratio)
        elif self.compression_type == "semantic":
            # Semantic compression
            from compression.semantic_compression import SemanticCompressor
            self.semantic_compressor = SemanticCompressor()
        else:
            # No compression
            self.compressor_1_2 = None
            self.compressor_2_3 = None
    
    def _compress_context(
        self,
        context: str,
        stage: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Compress context based on compression type and stage.
        
        Args:
            context: Context to compress
            stage: Which stage ("1_to_2" or "2_to_3")
            metadata: Optional metadata for compression
            
        Returns:
            Compressed context
        """
        if metadata is None:
            metadata = {}
        
        if self.compression_type == "none":
            return context
        
        if self.compression_type == "fixed":
            compressor = self.compressor_1_2 if stage == "1_to_2" else self.compressor_2_3
            if compressor:
                return compressor.compress(context)
            return context
        
        if self.compression_type in ("role_specific", "dynamic"):
            clarifier = self.clarifier_1_2 if stage == "1_to_2" else self.clarifier_2_3
            if clarifier:
                return clarifier.clarify(
                    context,
                    metadata=metadata,
                    target_compression=self.compression_ratio
                )
            return context
        
        if self.compression_type == "semantic":
            return self.semantic_compressor.compress_semantically(
                context,
                metadata=metadata,
                target_ratio=self.compression_ratio
            )
        
        return context

    def process(
        self,
        question: str,
        document: str,
        track_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Process a question through the agent chain.

        Args:
            question: Question to answer
            document: Source document/passage
            track_metrics: Whether to track detailed metrics

        Returns:
            Dictionary with final answer and metrics
        """
        start_time = time.time()
        results = {
            'question': question,
            'success': False,
            'error': None,
            'verification_skipped': False
        }

        try:
            # Agent 1: Retriever
            logger.info("=== Agent 1: Retriever ===")
            retriever_result = self.retriever.process(question, document)

            context_1 = retriever_result['output']
            logger.info(f"Retriever output: {len(context_1)} chars")

            # Compress for Agent 2
            context_1_compressed = self._compress_context(
                context_1, 
                stage="1_to_2",
                metadata={'question': question}
            )

            logger.info(
                f"Context 1→2: {len(context_1)} → {len(context_1_compressed)} chars"
            )

            # Agent 2: Reasoner
            logger.info("=== Agent 2: Reasoner ===")
            reasoner_result = self.reasoner.process(
                question,
                context_1_compressed
            )

            context_2 = reasoner_result['output']
            reasoner_confidence = reasoner_result.get('confidence', 0.0)
            logger.info(f"Reasoner output: {len(context_2)} chars (confidence: {reasoner_confidence:.2f})")

            # Check if we can skip verification based on confidence
            if (self.skip_verification_if_confident and 
                reasoner_confidence >= self.confidence_threshold):
                logger.info(f"Skipping verification (confidence {reasoner_confidence:.2f} >= {self.confidence_threshold})")
                final_answer = self.reasoner.extract_answer(context_2)
                results['verification_skipped'] = True
                verifier_result = {
                    'output': context_2,
                    'final_answer': final_answer,
                    'input_tokens': 0,
                    'output_tokens': 0
                }
                context_2_compressed = context_2
            else:
                # Compress for Agent 3
                context_2_compressed = self._compress_context(
                    context_2,
                    stage="2_to_3",
                    metadata={
                        'question': question,
                        'retrieval': context_1_compressed
                    }
                )

                logger.info(
                    f"Context 2→3: {len(context_2)} → {len(context_2_compressed)} chars"
                )

                # Agent 3: Verifier
                logger.info("=== Agent 3: Verifier ===")
                verifier_result = self.verifier.process(
                    question,
                    context_2_compressed
                )

            final_answer = verifier_result['final_answer']
            logger.info(f"Final answer: {final_answer[:100]}...")

            # Aggregate results
            end_time = time.time()
            results.update({
                'success': True,
                'final_answer': final_answer,
                'latency': end_time - start_time,
                'retriever_output': context_1,
                'reasoner_output': context_2,
                'verifier_output': verifier_result['output'],
                'context_sizes': {
                    'retriever': len(context_1.split()),
                    'retriever_compressed': len(context_1_compressed.split()),
                    'reasoner': len(context_2.split()),
                    'reasoner_compressed': len(context_2_compressed.split()),
                },
                'token_counts': {
                    'retriever_input': retriever_result['input_tokens'],
                    'retriever_output': retriever_result['output_tokens'],
                    'reasoner_input': reasoner_result['input_tokens'],
                    'reasoner_output': reasoner_result['output_tokens'],
                    'verifier_input': verifier_result['input_tokens'],
                    'verifier_output': verifier_result['output_tokens'],
                }
            })

            # Add memory info
            if track_metrics:
                results['memory_usage'] = self.verifier.get_memory_usage()

        except Exception as e:
            logger.error(f"Error in agent chain: {e}", exc_info=True)
            results['success'] = False
            results['error'] = str(e)
            results['latency'] = time.time() - start_time

        return results

    def cleanup(self):
        """Clean up all agents and free memory."""
        logger.info("Cleaning up agent chain...")
        self.retriever.cleanup()
        self.reasoner.cleanup()
        self.verifier.cleanup()

    def __repr__(self) -> str:
        return (
            f"AgentChain(model={self.model_name}, "
            f"compression={self.compression_type})"
        )
