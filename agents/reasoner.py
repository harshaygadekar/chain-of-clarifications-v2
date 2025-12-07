"""
Reasoner Agent - Second agent in the chain

The Reasoner's role is to:
1. Receive extracted information from the Retriever
2. Apply logical reasoning to the question
3. Generate a candidate answer with supporting reasoning
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)


class ReasonerAgent(BaseAgent):
    """
    Reasoner Agent - Applies logical reasoning to generate answers.

    This agent takes the extracted information from the Retriever
    and uses it to formulate an answer to the question.
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
        Initialize the Reasoner agent.

        Args:
            model_name: Hugging Face model identifier
            device: Computing device
            max_length: Maximum sequence length
            model: Pre-loaded model (optional, for sharing)
            tokenizer: Pre-loaded tokenizer (optional, for sharing)
        """
        super().__init__(
            role="reasoner",
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
        Construct reasoner-specific prompt.

        The reasoner needs to use the extracted information to
        formulate a logical answer.

        Args:
            question: The question to answer
            context: Extracted relevant information from Retriever
            **kwargs: Additional parameters

        Returns:
            Formatted prompt for the reasoner
        """
        prompt = f"""You are a reasoning specialist. Your task is to analyze the given information and provide a clear, logical answer to the question.

Question: {question}

Relevant Information:
{context}

Task:
1. Carefully read the question and the relevant information provided
2. Apply logical reasoning to connect the information to the question
3. Formulate a clear, concise answer
4. Provide your reasoning chain that led to this answer
5. If the information is insufficient, state what's missing

Your Analysis and Answer:"""

        return prompt

    def process(
        self,
        question: str,
        context: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process the extracted information and generate an answer.

        Args:
            question: The question to answer
            context: Relevant information from Retriever
            metadata: Additional metadata

        Returns:
            Dictionary containing:
                - output: Reasoning and answer
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
        logger.info(f"Reasoner processing question: {question[:100]}...")
        output = self.generate_response(
            prompt,
            max_new_tokens=min(450, self.max_length // 2)
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
            'memory_usage': self.get_memory_usage(),
            'confidence': self.compute_confidence(output)
        }

        logger.info(f"Reasoner generated {output_tokens} tokens of reasoning (confidence: {result['confidence']:.2f})")

        return result
    
    def compute_confidence(self, output: str) -> float:
        """
        Compute confidence score for the reasoning output.
        
        Confidence is based on:
        - Presence of uncertainty words (decreases)
        - Presence of certainty words (increases)
        - Answer clarity indicators
        - Output length (very short = low confidence)
        
        Args:
            output: The reasoning output text
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not output or output.startswith('[ERROR:'):
            return 0.0
        
        confidence = 0.5  # Base confidence
        output_lower = output.lower()
        
        # Uncertainty indicators (decrease confidence)
        uncertainty_words = [
            'maybe', 'perhaps', 'possibly', 'might', 'could be',
            'not sure', 'uncertain', 'unclear', 'don\'t know',
            'hard to say', 'difficult to determine', 'ambiguous',
            'insufficient information', 'cannot determine'
        ]
        for word in uncertainty_words:
            if word in output_lower:
                confidence -= 0.1
        
        # Certainty indicators (increase confidence)
        certainty_words = [
            'clearly', 'definitely', 'certainly', 'obviously',
            'the answer is', 'therefore', 'thus', 'hence',
            'confident', 'sure', 'evident', 'conclusively'
        ]
        for word in certainty_words:
            if word in output_lower:
                confidence += 0.08
        
        # Answer structure indicators (increase confidence)
        if 'answer:' in output_lower or 'conclusion:' in output_lower:
            confidence += 0.1
        
        # Very short output = low confidence
        word_count = len(output.split())
        if word_count < 10:
            confidence -= 0.2
        elif word_count > 50:
            confidence += 0.05
        
        # Contains specific entities/facts (increase confidence)
        import re
        numbers = len(re.findall(r'\b\d+\b', output))
        capitals = len(re.findall(r'\b[A-Z][a-z]+', output))
        if numbers > 0:
            confidence += 0.05
        if capitals > 2:
            confidence += 0.05
        
        # Clamp to valid range
        return max(0.0, min(1.0, confidence))

    def extract_answer(self, reasoning_output: str) -> str:
        """
        Extract the final answer from the reasoning output.

        Args:
            reasoning_output: The full reasoning text

        Returns:
            Extracted answer string
        """
        import re

        # Check for error markers first
        if reasoning_output.startswith("[ERROR:"):
            logger.error(f"Reasoner: Error in reasoning output: {reasoning_output}")
            return reasoning_output

        # Try to find explicit answer markers
        patterns = [
            r'Answer:\s*(.+?)(?:\n|$)',
            r'The answer is\s*(.+?)(?:\n|\.)',
            r'Therefore,?\s*(.+?)(?:\n|\.)',
            r'Conclusion:\s*(.+?)(?:\n|$)',
            r'Final answer:\s*(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, reasoning_output, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Validate extracted answer doesn't look like prompt
                if self._is_valid_answer(answer):
                    return answer

        # Improved fallback: use last meaningful sentence instead of first
        # (conclusions often come at the end)
        sentences = [s.strip() for s in reasoning_output.split('.') if s.strip()]

        # Try last sentences first
        for sent in reversed(sentences):
            if self._is_valid_answer(sent):
                return sent

        # If no valid sentence found, return truncated output with warning
        logger.warning(f"Reasoner: Could not extract clean answer, returning truncated output")
        return reasoning_output[:100].strip() if reasoning_output else "[ERROR: Empty reasoning output]"

    def _is_valid_answer(self, answer: str) -> bool:
        """
        Check if extracted answer is valid (not prompt-like).

        Args:
            answer: Candidate answer string

        Returns:
            True if answer appears valid
        """
        if not answer or len(answer) < 3:
            return False

        # Check if it looks like a prompt instruction
        prompt_indicators = [
            'your task is',
            'you are a',
            'carefully read',
            'apply logical',
            'formulate a',
            'provide your',
            'task:',
            'question:',
            'relevant information:',
        ]

        answer_lower = answer.lower()
        for indicator in prompt_indicators:
            if indicator in answer_lower:
                return False

        return True
