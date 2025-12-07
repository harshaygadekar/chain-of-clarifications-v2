"""
Conversation Chain for Multi-Turn QA

Extends AgentChain to support multi-turn conversations with
context carryover between turns.
"""

from typing import Dict, List, Optional, Any
from agents.agent_chain import AgentChain
import logging

logger = logging.getLogger(__name__)


class ConversationChain(AgentChain):
    """
    Agent chain with conversation history support.
    
    Maintains context across multiple question turns,
    enabling follow-up questions and anaphora resolution.
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None,
        compression_type: str = "none",
        compression_ratio: float = 0.5,
        max_history_turns: int = 5
    ):
        """
        Initialize conversation chain.
        
        Args:
            model_name: Model to use for all agents
            device: Computing device
            compression_type: Type of compression
            compression_ratio: Compression ratio
            max_history_turns: Maximum turns to keep in history
        """
        super().__init__(
            model_name=model_name,
            device=device,
            compression_type=compression_type,
            compression_ratio=compression_ratio
        )
        
        self.max_history_turns = max_history_turns
        self.conversation_history: List[Dict] = []
        self.current_document: Optional[str] = None
    
    def start_conversation(self, document: str):
        """
        Start a new conversation with a document.
        
        Args:
            document: Source document for the conversation
        """
        self.reset_conversation()
        self.current_document = document
        logger.info("Started new conversation")
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.current_document = None
        logger.info("Conversation history cleared")
    
    def process_turn(
        self,
        question: str,
        document: Optional[str] = None,
        track_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single conversation turn.
        
        Args:
            question: Current question
            document: Optional document (uses stored if None)
            track_metrics: Whether to track metrics
            
        Returns:
            Result dictionary with answer and metadata
        """
        # Use provided document or stored one
        if document is not None:
            self.current_document = document
        
        if self.current_document is None:
            return {
                'success': False,
                'error': 'No document provided for conversation',
                'final_answer': ''
            }
        
        # Resolve references in question using history
        resolved_question = self._resolve_references(question)
        
        # Create augmented context with history
        augmented_context = self._build_context_with_history()
        
        # Process through agent chain
        result = self.process(
            question=resolved_question,
            document=augmented_context,
            track_metrics=track_metrics
        )
        
        # Store in history if successful
        if result.get('success', False):
            self._add_to_history(question, result.get('final_answer', ''))
        
        # Add conversation metadata
        result['original_question'] = question
        result['resolved_question'] = resolved_question
        result['turn_number'] = len(self.conversation_history)
        
        return result
    
    def _resolve_references(self, question: str) -> str:
        """
        Resolve pronouns and references using conversation history.
        
        Args:
            question: Current question with potential references
            
        Returns:
            Question with references resolved
        """
        if not self.conversation_history:
            return question
        
        resolved = question
        
        # Get the last mentioned entity/answer
        last_answer = self.conversation_history[-1].get('answer', '')
        last_question = self.conversation_history[-1].get('question', '')
        
        # Simple reference resolution patterns
        # Replace pronouns with last answer if it makes sense
        pronoun_patterns = [
            (r'\b[Ii]t\b', last_answer),
            (r'\b[Tt]hey\b', last_answer),
            (r'\b[Tt]his\b', last_answer),
            (r'\b[Tt]hat\b', last_answer),
            (r'\b[Hh]e\b', last_answer),
            (r'\b[Ss]he\b', last_answer),
            (r'\b[Tt]hem\b', last_answer),
        ]
        
        import re
        for pattern, replacement in pronoun_patterns:
            # Only replace if answer looks like a valid entity
            if replacement and len(replacement.split()) <= 5:
                # Check if pattern is at start or after common question words
                if re.search(rf'(^|when|where|what|how)(.*?){pattern}', resolved, re.IGNORECASE):
                    resolved = re.sub(pattern, replacement, resolved, count=1)
                    logger.info(f"Resolved reference: '{pattern}' -> '{replacement}'")
                    break
        
        return resolved
    
    def _build_context_with_history(self) -> str:
        """
        Build context that includes relevant conversation history.
        
        Returns:
            Augmented context string
        """
        parts = []
        
        # Add document
        parts.append(f"Document:\n{self.current_document}")
        
        # Add relevant history
        if self.conversation_history:
            parts.append("\nConversation History:")
            for i, turn in enumerate(self.conversation_history[-self.max_history_turns:]):
                q = turn.get('question', '')
                a = turn.get('answer', '')
                parts.append(f"Q{i+1}: {q}")
                parts.append(f"A{i+1}: {a}")
        
        return '\n'.join(parts)
    
    def _add_to_history(self, question: str, answer: str):
        """
        Add a turn to conversation history.
        
        Args:
            question: The question asked
            answer: The answer provided
        """
        self.conversation_history.append({
            'question': question,
            'answer': answer
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]
    
    def get_conversation_summary(self) -> str:
        """
        Get a summary of the current conversation.
        
        Returns:
            Conversation summary string
        """
        if not self.conversation_history:
            return "No conversation history"
        
        lines = [f"Conversation ({len(self.conversation_history)} turns):"]
        for i, turn in enumerate(self.conversation_history):
            lines.append(f"  Turn {i+1}:")
            lines.append(f"    Q: {turn['question'][:50]}...")
            lines.append(f"    A: {turn['answer'][:50]}...")
        
        return '\n'.join(lines)


def run_conversation_demo():
    """Demo of multi-turn conversation."""
    print("=" * 60)
    print("Multi-Turn Conversation Demo")
    print("=" * 60)
    
    # Sample document
    document = """
    Albert Einstein was born in Ulm, Germany in 1879. He developed the theory 
    of relativity, one of the two pillars of modern physics. Einstein received 
    the Nobel Prize in Physics in 1921 for his explanation of the photoelectric 
    effect. He later moved to the United States in 1933 and worked at the 
    Institute for Advanced Study in Princeton until his death in 1955.
    """
    
    # Initialize conversation chain
    conv_chain = ConversationChain(compression_type="role_specific")
    
    # Start conversation
    conv_chain.start_conversation(document)
    
    # Multi-turn questions
    questions = [
        "Where was Einstein born?",
        "When did he win the Nobel Prize?",  # "he" refers to Einstein
        "What was it for?",  # "it" refers to Nobel Prize
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        result = conv_chain.process_turn(q)
        if result['success']:
            print(f"A: {result['final_answer']}")
            if q != result.get('resolved_question'):
                print(f"  (Resolved: {result['resolved_question']})")
        else:
            print(f"Error: {result.get('error', 'Unknown')}")
    
    # Print conversation summary
    print("\n" + conv_chain.get_conversation_summary())
    
    # Cleanup
    conv_chain.cleanup()


if __name__ == "__main__":
    run_conversation_demo()
