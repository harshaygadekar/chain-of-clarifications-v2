"""
Base Agent Class for Chain of Clarifications System

This module provides the base class for all agents in the multi-agent chain.
Each agent performs a specific role: Retrieval, Reasoning, or Verification.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for all agents in the chain.

    Attributes:
        role (str): The role of this agent (retriever, reasoner, verifier)
        model_name (str): Name of the language model to use
        device (str): Device to run the model on (cuda/cpu)
        max_length (int): Maximum sequence length for generation
    """

    def __init__(
        self,
        role: str,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize the base agent.

        Args:
            role: Role identifier for this agent
            model_name: Hugging Face model identifier
            device: Computing device (auto-detected if None)
            max_length: Maximum tokens for generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            model: Pre-loaded model (optional, for sharing across agents)
            tokenizer: Pre-loaded tokenizer (optional, for sharing across agents)
        """
        self.role = role
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p

        # Auto-detect device with detailed diagnostics
        if device is None:
            # Only log diagnostics for first agent (when loading fresh)
            if model is None:
                self._log_gpu_diagnostics()
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"CUDA is available - using GPU")
            else:
                self.device = "cpu"
                logger.warning(f"CUDA not available - falling back to CPU")
        else:
            self.device = device
            logger.info(f"Using explicitly specified device: {self.device}")

        logger.info(f"Initializing {self.role} agent on {self.device}")

        # Use pre-loaded model/tokenizer if provided, otherwise load new
        self.tokenizer = tokenizer
        self.model = model
        self.is_seq2seq = False  # Track if model is seq2seq or causal

        if self.model is None or self.tokenizer is None:
            logger.info(f"{self.role}: Loading new model")
            self._load_model()
        else:
            logger.info(f"{self.role}: Using shared pre-loaded model")
            # Detect model type from model class
            model_class_name = self.model.__class__.__name__
            self.is_seq2seq = 'Seq2Seq' in model_class_name or 'T5' in model_class_name
            logger.info(f"{self.role}: Detected model type - seq2seq={self.is_seq2seq}")

    def _log_gpu_diagnostics(self):
        """Log detailed GPU diagnostics for debugging."""
        logger.info("=" * 60)
        logger.info("GPU DIAGNOSTICS")
        logger.info("=" * 60)

        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA Available: {cuda_available}")

        if cuda_available:
            # GPU count and details
            gpu_count = torch.cuda.device_count()
            logger.info(f"GPU Count: {gpu_count}")

            # Current device
            current_device = torch.cuda.current_device()
            logger.info(f"Current Device ID: {current_device}")

            # Device properties
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}")
                logger.info(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
                logger.info(f"  CUDA Capability: {props.major}.{props.minor}")

                # Memory stats
                try:
                    allocated = torch.cuda.memory_allocated(i) / 1024**2
                    reserved = torch.cuda.memory_reserved(i) / 1024**2
                    logger.info(f"  Memory Allocated: {allocated:.2f} MB")
                    logger.info(f"  Memory Reserved: {reserved:.2f} MB")
                except Exception as e:
                    logger.warning(f"  Could not get memory stats: {e}")

            # CUDA version
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
        else:
            logger.warning("CUDA not available - GPU diagnostics skipped")
            logger.info(f"PyTorch Version: {torch.__version__}")

        logger.info("=" * 60)

    def _load_model(self):
        """Load the language model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Detect model type and load accordingly
            # T5 and FLAN-T5 models are seq2seq, GPT models are causal
            is_t5_family = any(x in self.model_name.lower() for x in ['t5', 'flan'])

            # Determine dtype based on device
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            logger.info(f"Using dtype: {dtype}")

            if is_t5_family:
                logger.info(f"Detected Seq2Seq model (T5 family)")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype
                )
                self.is_seq2seq = True
            else:
                logger.info(f"Detected CausalLM model")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype
                )
                self.is_seq2seq = False

            # Move model to device with error handling
            try:
                if self.device == "cuda":
                    # Force CUDA device selection
                    cuda_device = torch.device("cuda:0")
                    logger.info(f"Moving model to {cuda_device}")
                    self.model.to(cuda_device)

                    # Log GPU memory allocation after model loading
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated(0) / 1024**2
                        reserved = torch.cuda.memory_reserved(0) / 1024**2
                        logger.info(f"GPU Memory after model load:")
                        logger.info(f"  Allocated: {allocated:.2f} MB")
                        logger.info(f"  Reserved: {reserved:.2f} MB")
                else:
                    self.model.to(self.device)
                    logger.info(f"Model loaded on CPU")

            except RuntimeError as e:
                logger.error(f"Failed to move model to {self.device}: {e}")
                logger.warning(f"Falling back to CPU")
                self.device = "cpu"
                self.model.to(self.device)

            self.model.eval()

            # Set pad token if not exists (mainly for CausalLM models)
            if self.tokenizer.pad_token is None:
                if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = self.tokenizer.unk_token

            logger.info(f"Model loaded successfully for {self.role} (seq2seq={self.is_seq2seq}, device={self.device})")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        """
        Generate a response using the language model.

        Args:
            prompt: Input prompt for the model
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.device)

            # Generate response with model-specific parameters
            with torch.no_grad():
                if self.is_seq2seq:
                    # Seq2Seq models (T5/FLAN-T5) - don't pass pad_token_id to avoid warnings
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        **kwargs
                    )
                    # For seq2seq, decode the entire output (no input slicing needed)
                    response = self.tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True
                    )
                else:
                    # CausalLM models (GPT-2, etc.)
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        **kwargs
                    )
                    # For causal models, skip the input tokens in output
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )

            # Validation after decoding
            response = response.strip()

            # Check if response is empty
            if not response:
                logger.warning(f"{self.role}: Generated empty response")
                return "[ERROR: Empty response generated]"

            # Check if response is suspiciously short
            if len(response) < 10:
                logger.warning(f"{self.role}: Generated very short response ({len(response)} chars): '{response}'")

            # Check for prompt contamination (response starts with part of prompt)
            # Only check for causal models where this is more common
            if not self.is_seq2seq:
                prompt_start = prompt[:50].lower().strip()
                response_start = response[:50].lower().strip()
                if response_start and prompt_start and response_start.startswith(prompt_start[:20]):
                    logger.warning(f"{self.role}: Detected prompt contamination in response")
                    return "[ERROR: Response contaminated with prompt text]"

            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.

        Args:
            text: Input text to count

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def process(
        self,
        question: str,
        context: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process input and generate output.
        This method should be overridden by child classes.

        Args:
            question: The question to answer
            context: Context information from previous agents
            metadata: Additional metadata

        Returns:
            Dictionary containing output and metadata
        """
        raise NotImplementedError("Subclasses must implement process()")

    def get_prompt(
        self,
        question: str,
        context: str,
        **kwargs
    ) -> str:
        """
        Construct the prompt for this agent.
        Should be overridden by child classes.

        Args:
            question: The question
            context: Context information
            **kwargs: Additional parameters

        Returns:
            Formatted prompt string
        """
        raise NotImplementedError("Subclasses must implement get_prompt()")

    def cleanup(self):
        """Clean up GPU memory."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"{self.role} agent cleaned up")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.

        Returns:
            Dictionary with memory statistics
        """
        memory_info = {}

        if torch.cuda.is_available():
            memory_info['allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            memory_info['reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            memory_info['max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024**2
        else:
            memory_info['allocated_mb'] = 0
            memory_info['reserved_mb'] = 0
            memory_info['max_allocated_mb'] = 0

        return memory_info

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(role={self.role}, model={self.model_name}, device={self.device})"
