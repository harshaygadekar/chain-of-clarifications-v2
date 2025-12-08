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
        # Check if this is a summarization task
        is_summarization = 'summarize' in question.lower() or 'summary' in question.lower()
        
        if is_summarization:
            prompt = f"""Write a 2-3 sentence summary based on the analysis below.

Example 1:
Analysis: "The paper proposes a novel attention mechanism for transformer models. Experiments on GLUE benchmark show 2.3% improvement over BERT. The method reduces computational cost by 40%."
Summary: The paper introduces a new attention mechanism that improves transformer performance by 2.3% on GLUE while reducing computational cost by 40%.

Example 2:
Analysis: "Researchers evaluated three compression methods on scientific documents. Role-specific compression achieved the best F1 score of 0.78. Fixed-ratio compression was fastest but lost semantic information."
Summary: Role-specific compression achieved the highest F1 score (0.78) among three methods tested on scientific documents, outperforming fixed-ratio compression which sacrificed semantic accuracy for speed.

Now write a summary for this analysis:

Analysis:
{context}

Summary:"""
        else:
            prompt = f"""Answer the question based on the scientific analysis provided.

Example:
Question: What dataset was used to evaluate the model?
Analysis: "The authors evaluated their approach on SQuAD 2.0 and Natural Questions datasets. Results showed significant improvements on both benchmarks."
Answer: The model was evaluated on SQuAD 2.0 and Natural Questions datasets.

Now answer this question directly:

Question: {question}

Analysis:
{context}

Answer:"""

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

        # Clean junk prefixes from output
        cleaned_output = self._clean_output(verification_output)

        # Try to find explicit answer markers in cleaned output
        patterns = [
            r'Final Answer:\s*(.+?)(?:\n|$)',
            r'Answer:\s*(.+?)(?:\n|$)',
            r'Verified Answer:\s*(.+?)(?:\n|$)',
            r'The answer is\s*(.+?)(?:\n|\.)',
            r'Conclusion:\s*(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned_output, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Validate extracted answer doesn't look like prompt
                if self._is_valid_answer(answer):
                    return self._clean_output(answer)

        # Improved fallback: use last meaningful sentence instead of first
        # (final answer often comes at the end)
        sentences = [s.strip() for s in cleaned_output.split('.') if s.strip()]

        # Try last sentences first
        for sent in reversed(sentences):
            if self._is_valid_answer(sent):
                return self._clean_output(sent)

        # If no valid sentence found, return cleaned output
        logger.warning("Verifier: Could not extract clean answer, returning cleaned output")
        return cleaned_output[:200].strip() if len(cleaned_output) > 200 else cleaned_output

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

        # Check if it looks like a prompt instruction or junk output
        junk_indicators = [
            'your task is',
            'you are a',
            'review the',
            'check if',
            'identify any',
            'provide your',
            'task:',
            'question:',
            'passage:',
            'confidence:',
            'high',
            'medium',
            'low',
            'answer:',
            'summary:',
        ]

        answer_lower = answer.lower().strip()
        
        # Check if answer IS just a junk word
        if answer_lower in ['high', 'medium', 'low', 'yes', 'no', 'true', 'false']:
            return False
        
        for indicator in junk_indicators:
            if answer_lower.startswith(indicator):
                return False

        return True
    
    def _clean_output(self, text: str) -> str:
        """
        Clean junk prefixes from model output.
        
        Args:
            text: Raw model output
            
        Returns:
            Cleaned text with junk prefixes removed
        """
        import re
        
        # Patterns to remove from start of output
        junk_patterns = [
            r'^Confidence:\s*(High|Medium|Low)\s*\.?\s*',
            r'^(High|Medium|Low)\s*\.?\s*',
            r'^A\.\s*',
            r'^B\.\s*',
            r'^C\.\s*',
            r'^D\.\s*',
            r'^Answer:\s*',
            r'^Summary:\s*',
            r'^Final Answer:\s*',
            r'^The answer is:\s*',
            r'^\d+\.\s*',
        ]
        
        cleaned = text.strip()
        
        # Apply each pattern
        for pattern in junk_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        
        # If we stripped everything, return original
        if len(cleaned) < 10 and len(text) > 20:
            return text.strip()
        
        return cleaned

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
