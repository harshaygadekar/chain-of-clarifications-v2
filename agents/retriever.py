"""
Retriever Agent - First agent in the chain

The Retriever's role is to:
1. Read the document/passage
2. Identify relevant information for answering the question
3. Extract and organize key facts, entities, and evidence
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseAgent):
    """
    Retriever Agent - Identifies and extracts relevant information.

    This agent focuses on finding answer-bearing passages and extracting
    relevant entities, facts, and context for the next agent.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None,
        max_length: int = 1024,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize the Retriever agent.

        Args:
            model_name: Hugging Face model identifier
            device: Computing device
            max_length: Maximum sequence length
            model: Pre-loaded model (optional, for sharing)
            tokenizer: Pre-loaded tokenizer (optional, for sharing)
        """
        super().__init__(
            role="retriever",
            model_name=model_name,
            device=device,
            max_length=max_length,
            model=model,
            tokenizer=tokenizer
        )

    def get_prompt(
        self,
        question: str,
        context: str,
        **kwargs
    ) -> str:
        """
        Construct retriever-specific prompt.

        The retriever needs to extract relevant information from the document
        that could help answer the question.

        Args:
            question: The question to answer
            context: The full document/passage
            **kwargs: Additional parameters

        Returns:
            Formatted prompt for the retriever
        """
        prompt = f"""You are a document retrieval specialist. Your task is to read the given passage and extract the most relevant information that could help answer the question.

Question: {question}

Passage:
{context}

Task:
1. Identify key sentences that contain information relevant to the question
2. Extract important entities (names, dates, numbers, locations)
3. Note any facts that might support answering the question
4. Organize the relevant information clearly

Extracted Relevant Information:"""

        return prompt

    def process(
        self,
        question: str,
        context: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process the document and extract relevant information.

        Args:
            question: The question to answer
            context: The full document/passage
            metadata: Additional metadata

        Returns:
            Dictionary containing:
                - output: Extracted relevant information
                - input_tokens: Number of input tokens
                - output_tokens: Number of output tokens
                - role: Agent role
        """
        if metadata is None:
            metadata = {}

        # Create the prompt
        prompt = self.get_prompt(question, context)

        # Count input tokens
        input_tokens = self.count_tokens(prompt)

        # Generate response
        logger.info(f"Retriever processing question: {question[:100]}...")
        output = self.generate_response(
            prompt,
            max_new_tokens=min(400, self.max_length // 2)
        )

        # Count output tokens
        output_tokens = self.count_tokens(output)

        result = {
            'output': output,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'role': self.role,
            'question': question,
            'memory_usage': self.get_memory_usage()
        }

        logger.info(f"Retriever extracted {output_tokens} tokens of relevant info")

        return result

    def extract_entities(self, text: str) -> Dict[str, list]:
        """
        Extract entities from text (simplified version).

        Args:
            text: Text to extract entities from

        Returns:
            Dictionary with entity types and values
        """
        # This is a simplified version - in practice, could use NER models
        entities = {
            'numbers': [],
            'years': [],
            'names': [],
            'locations': []
        }

        # Simple pattern matching for years (1900-2099)
        import re
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        entities['years'] = list(set(years))

        # Extract capitalized words (potential names/locations)
        words = text.split()
        capitalized = [w for w in words if w and w[0].isupper() and len(w) > 1]
        entities['names'] = list(set(capitalized[:10]))  # Limit to top 10

        return entities
