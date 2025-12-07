"""
Verifier Agent - Third agent in the chain

The Verifier's role is to:
1. Receive the answer and reasoning from the Reasoner
2. Verify the correctness and consistency
3. Produce a final, validated answer
"""

from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)


class VerifierAgent(BaseAgent):
    """
    Verifier Agent - Validates and refines the answer.

    This agent checks the reasoning and answer from the Reasoner,
    verifies consistency, and produces the final answer.
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
        Initialize the Verifier agent.

        Args:
            model_name: Hugging Face model identifier
            device: Computing device
            max_length: Maximum sequence length
            model: Pre-loaded model (optional, for sharing)
            tokenizer: Pre-loaded tokenizer (optional, for sharing)
        """
        super().__init__(
            role="verifier",
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
        Construct verifier-specific prompt.

        The verifier needs to check the reasoning and validate the answer.

        Args:
            question: The original question
            context: Reasoning and answer from Reasoner
            **kwargs: Additional parameters

        Returns:
            Formatted prompt for the verifier
        """
        prompt = f"""You are a verification specialist. Your task is to review the reasoning and answer provided, check for logical consistency, and produce a final validated answer.

Question: {question}

Proposed Reasoning and Answer:
{context}

Task:
1. Review the reasoning chain - is it logical and consistent?
2. Check if the answer actually addresses the question
3. Identify any potential errors or inconsistencies
4. Provide your final verified answer (or corrected answer if needed)
5. Rate your confidence (High/Medium/Low)

Your Verification and Final Answer:"""

        return prompt

    def process(
        self,
        question: str,
        context: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Verify the reasoning and produce final answer.

        Args:
            question: The original question
            context: Reasoning from Reasoner
            metadata: Additional metadata

        Returns:
            Dictionary containing:
                - output: Verification and final answer
                - final_answer: Extracted final answer
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
        logger.info(f"Verifier processing question: {question[:100]}...")
        output = self.generate_response(
            prompt,
            max_new_tokens=min(300, self.max_length // 2)
        )

        # Count output tokens
        output_tokens = self.count_tokens(output)

        # Extract final answer
        final_answer = self.extract_final_answer(output)

        result = {
            'output': output,
            'final_answer': final_answer,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'role': self.role,
            'question': question,
            'memory_usage': self.get_memory_usage()
        }

        logger.info(f"Verifier produced final answer: {final_answer[:50]}...")

        return result

    def extract_final_answer(self, verification_output: str) -> str:
        """
        Extract the final answer from verification output.

        Args:
            verification_output: The full verification text

        Returns:
            Extracted final answer
        """
        import re

        # Check for error markers first
        if verification_output.startswith("[ERROR:"):
            logger.error(f"Verifier: Error in verification output: {verification_output}")
            return verification_output

        # Try to find explicit answer markers
        patterns = [
            r'Final Answer:\s*(.+?)(?:\n|$)',
            r'Answer:\s*(.+?)(?:\n|$)',
            r'Verified Answer:\s*(.+?)(?:\n|$)',
            r'The answer is\s*(.+?)(?:\n|\.)',
            r'Conclusion:\s*(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, verification_output, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Validate extracted answer doesn't look like prompt
                if self._is_valid_answer(answer):
                    return answer

        # Improved fallback: use last meaningful sentence instead of first
        # (final answer often comes at the end)
        sentences = [s.strip() for s in verification_output.split('.') if s.strip()]

        # Try last sentences first
        for sent in reversed(sentences):
            if self._is_valid_answer(sent):
                return sent

        # If no valid sentence found, return truncated output with warning
        logger.warning(f"Verifier: Could not extract clean answer, returning truncated output")
        return verification_output[:100].strip()

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
            'review the',
            'check if',
            'identify any',
            'provide your',
            'task:',
            'question:',
            'passage:',
        ]

        answer_lower = answer.lower()
        for indicator in prompt_indicators:
            if indicator in answer_lower:
                return False

        return True

    def extract_confidence(self, verification_output: str) -> str:
        """
        Extract confidence level from verification.

        Args:
            verification_output: The verification text

        Returns:
            Confidence level (High/Medium/Low/Unknown)
        """
        import re

        # Look for confidence indicators
        confidence_pattern = r'Confidence:\s*(High|Medium|Low)'
        match = re.search(confidence_pattern, verification_output, re.IGNORECASE)

        if match:
            return match.group(1).capitalize()

        # Heuristic: check for uncertainty markers
        if any(word in verification_output.lower() for word in ['uncertain', 'unclear', 'maybe', 'possibly']):
            return 'Low'
        elif any(word in verification_output.lower() for word in ['confident', 'certain', 'clearly', 'definitely']):
            return 'High'
        else:
            return 'Medium'
