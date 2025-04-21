import os
import json
import time
import random
import pandas as pd
import numpy as np
import uuid
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import re
import concurrent.futures
from tqdm import tqdm

# API clients
from openai import OpenAI
from anthropic import Anthropic

# ================ HARDCODED API KEYS ================
# Replace these with your actual API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# ====================================================

class TwentyQuestionsGame:
    """
    Implements a 20 Questions game where AI models play against each other.
    One model acts as the Answerer (has an entity in mind), and another as the Questioner (tries to guess).
    """
    
    # Categories and sample entities - expand these as needed
    ENTITIES = {
        "complex": [
             "burning man temple",
            "dark matter gravitational lensing", "existential nihilism",
            "neuromorphic computing", "symbiotic mutualism",
            "geopolitical sovereignty", "epistemological relativism",
            "recursive neural networks", "post-colonial discourse",
            "quantum entanglement", "cultural diaspora", 
            "blockchain technology", "collective unconscious",
        ],

        "simple": [
            "apple", "dog", "sun", "book", "car", 
            "chair", "tree", "water", "house", "pencil",
            "phone", "table", "bird", "shoe", "ball"
        ],
        
        "medium": [
            "electric guitar", "cryptocurrency", "antibiotics",
            "satellite dish", "virtual reality", "wind turbine",
            "espresso machine", "democracy", "black hole",
            "climate change", "photosynthesis", "vaccination"
        ],
        
        "people": [
            "Albert Einstein", "Beethoven", "Cleopatra", 
            "Nelson Mandela", "Marie Curie", "Shakespeare",
            "Leonardo da Vinci", "Muhammad Ali", "Frida Kahlo"
        ],
        
        "places": [
            "Mount Everest", "Amazon Rainforest", "Great Barrier Reef",
            "Eiffel Tower", "Grand Canyon", "Taj Mahal",
            "Antarctica", "Sahara Desert", "Great Wall of China"
        ],
        
        "fictional": [
            "Sherlock Holmes", "Hogwarts School", "lightsaber",
            "Atlantis", "unicorn", "Darth Vader",
            "Gandalf", "Narnia", "Batmobile"
        ]
    }
    
    def __init__(
        self,
        questioner_provider: str,
        questioner_model: str,
        answerer_provider: str,
        answerer_model: str,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        max_questions: int = 20,
        output_dir: str = "results",
        verbose: bool = True
    ):
        """
        Initialize a 20 Questions game.
        
        Args:
            questioner_provider: API provider for questioner ("openai" or "anthropic")
            questioner_model: Model name for questioner
            answerer_provider: API provider for answerer ("openai" or "anthropic")
            answerer_model: Model name for answerer
            openai_api_key: OpenAI API key (if None, uses OPENAI_API_KEY environment variable)
            anthropic_api_key: Anthropic API key (if None, uses ANTHROPIC_API_KEY environment variable)
            max_questions: Maximum number of questions allowed
            output_dir: Directory to save results
            verbose: Whether to print game progress
        """
        self.questioner_provider = questioner_provider
        self.questioner_model = questioner_model
        self.answerer_provider = answerer_provider
        self.answerer_model = answerer_model
        self.max_questions = max_questions
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Create clients with explicit API keys
        if questioner_provider == "openai":
            self.questioner_client = OpenAI(api_key=openai_api_key)
        elif questioner_provider == "anthropic":
            self.questioner_client = Anthropic(api_key=anthropic_api_key)
        
        if answerer_provider == "openai":
            self.answerer_client = OpenAI(api_key=openai_api_key)
        elif answerer_provider == "anthropic":
            self.answerer_client = Anthropic(api_key=anthropic_api_key)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Game state
        self.entity = None
        self.category = None
        self.questions_asked = []
        self.answers = []
        self.final_guess = None
        self.success = False
        self.question_count = 0
        
        # Store conversation messages with thinking blocks
        self.questioner_messages = []
        
        # Store thinking content for logging
        self.questioner_thinking = []
        self.answerer_thinking = []
        
        # Token usage tracking
        self.questioner_token_usage = []
        self.answerer_token_usage = []
    
    def select_entity(self, category: Optional[str] = None) -> Tuple[str, str]:
        """
        Select a random entity, optionally from a specific category.
        
        Args:
            category: Optional category to select from
            
        Returns:
            Tuple of (entity, category)
        """
        if category is None:
            category = random.choice(list(self.ENTITIES.keys()))
        
        entity = random.choice(self.ENTITIES[category])
        return entity, category
    
    def get_questioner_prompt(self) -> str:
        """
        Generate the prompt for the questioner model.
        
        Returns:
            Prompt string
        """
        # Determine game name based on max_questions
        game_name = "20 Questions"
        if self.max_questions == 30:
            game_name = "30 Questions" 
        elif self.max_questions != 20:
            game_name = f"{self.max_questions} Questions"
        
        base_prompt = (
            f"Let's play {game_name}, a deduction game where you must identify what the I am thinking of through strategic yes/no questions. They could be thinking of absolutely anything.\n\n"
            "GAMEPLAY STRATEGY:\n"
            "1. START WITH FUNDAMENTAL DIVISIONS: Begin with the broadest possible questions that divide the universe of possibilities (e.g., 'Is it physical/tangible?', 'Is it alive/organic?', 'Is it something that exists in the present day?')\n\n"
            "2. USE BINARY TREE THINKING:  Each question should efficiently split your remaining search space roughly in half. Example sequence:\n"
        "   - Is it physical/tangible? → Yes\n"
        "   - Is it alive/organic? → No\n"
        "   - Is it manufactured by humans? → Yes\n"
        "   - Is it primarily used indoors? → Yes\n\n"
            "3. ASK SPECIFIC, CLEAR QUESTIONS that can be answered with Yes, or No. Assume 'maybe' means the answerer cannot choose between yes and no."
            "4. AVOID THESE MISTAKES:\n"
            "   - Don't list multiple examples in one question (e.g., 'Is it a dog, cat, or horse?')\n"
            "   - Don't assume any category or domain initially - it could be from any domain\n"
            "   - Don't waste questions on low-information details before establishing basic categories\n\n"
            "   - Don't go down rabbit holes - if an answer doesn't seem to be narrowing the possibilities, change direction\n\n"
            "5. TRACK YOUR PROGRESS: Build a mental model of what you know and don't know\n\n"
            "6. BE DECISIVE AND EFFICIENT: If you're reasonably confident (70% or more) about the answer, make your guess! Don't waste questions just to be certain. Guessing correctly with fewer questions is better than using all 20 questions.\n\n"
            "7. NAVIGATE UNCERTAINTY: Sometimes the answerer may give inconsistent or slightly misleading answers due to different interpretations or misconceptions. If you suspect you're going down a wrong path:\n"
            "   - Consider asking a clarifying question from a different angle\n"
            "   - Be willing to pivot your line of questioning if you hit dead ends\n"
            "   - Look for patterns in the answers rather than treating each as absolute truth\n"
            "   - If answers seem contradictory, trust the overall pattern more than any single answer\n\n"
            "8. MAKE YOUR GUESS: When you're confident (even if not 100% certain), state '<guess>YOUR GUESS</guess>'. Making an educated guess early is better than using all your questions!\n\n"
            "IMPORTANT: Format your question ONLY using <question>YOUR QUESTION</question> tags. Any text outside these tags will be ignored."
            "Remember: The most efficient questions eliminate half the remaining possibilities. Think information theory."
        )
        
        # Add conversation history
        if self.questions_asked:
            base_prompt += "\n\nHere's the conversation so far:\n\n"
            for i, (question, answer) in enumerate(zip(self.questions_asked, self.answers)):
                base_prompt += f"Question {i+1}: {question}\n"
                base_prompt += f"Answer: {answer}\n\n"
            
            # Add question count reminder
            remaining = self.max_questions - self.question_count
            base_prompt += f"You have asked {self.question_count} questions. You have {remaining} questions remaining.\n\n"
            
            # Add explicit encouragement to guess earlier if we've asked several questions
            if self.question_count >= 8:
                base_prompt += "IMPORTANT: Don't hesitate to make a guess if you're reasonably confident! Efficiency is valued in this game.\n\n"
        
        base_prompt += "Ask your next question. If you're ready to guess, say '<guess>YOUR GUESS</guess>'."
        return base_prompt
    
    def get_answerer_prompt(self) -> str:
        """
        Generate the prompt for the answerer model.
        
        Returns:
            Prompt string
        """
        base_prompt = (
            f"You are playing a game of {self.max_questions} Questions. You are thinking of '{self.entity}'. The other player will ask yes/no questions to try to guess what you're thinking of."
            f"\n\nIMPORTANT: Be factually accurate in all your answers. Remember key facts about '{self.entity}'."
            f"\n\nRespond with ONLY one of these formats:"
            f"\n- <answer>yes</answer> (if the answer is affirmative)"
            f"\n- <answer>no</answer> (if the answer is negative)"
            f"\n- <answer>maybe</answer> (if you're uncertain, cannot choose between yes or no, or feel that either answer could be misleading to the questioner)"
            f"\n- <answer>correct</answer> (if they've guessed {self.entity})"
            f"\nYou may add a brief factual clarification AFTER your tagged answer when necessary for accuracy."
            f"\nIf the player mentions exactly what you're thinking of in their question, respond with '<answer>correct</answer> You've guessed {self.entity}' and the game will end immediately.\n\n"
        )
        
        # Add conversation history
        if self.questions_asked:
            base_prompt += "Here's the conversation so far:\n\n"
            for i, (question, answer) in enumerate(zip(self.questions_asked, self.answers)):
                base_prompt += f"Question {i+1}: {question}\n"
                base_prompt += f"Answer: {answer}\n\n"
        
        current_question = self.questions_asked[-1] if self.questions_asked else "No questions asked yet."
        base_prompt += f"The current question is: {current_question}\n\n"
        base_prompt += "Provide your answer (Yes/No/Maybe with minimal clarification if needed):"
        
        return base_prompt
    
    def get_model_response(self, provider: str, model: str, prompt: str, client, is_questioner: bool) -> str:
        """
        Get a response from an AI model using the appropriate API.
        
        Args:
            provider: API provider ("openai" or "anthropic")
            model: Model name
            prompt: Input prompt
            client: API client
            is_questioner: Whether this is for the questioner role
            
        Returns:
            Model response
        """
        # Set max_tokens based on model and role
        if is_questioner:
            #  higher max_tokens for questioners, especially with reasoning
            if model == "claude-3-5-sonnet-20241022":
                max_tokens = 8192
            else:
                max_tokens = 64000
        else:
            max_tokens = 5000  # Increased from 50 to give the answerer more room for accurate responses
        
        if self.verbose and False:  # Disable verbose API logging but keep the code
            role = "Questioner" if is_questioner else "Answerer"
            print(f"\n--- {role} API Request ({provider} - {model}) ---")
            print(f"Prompt: {prompt}")
        
        try:
            if provider == "openai":
                if self.verbose and False:  # Disable verbose API logging
                    print(f"Calling OpenAI API with model: {model}, max_tokens: {max_tokens}")
                
                # Try with current API parameter
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        response_format={"type": "text"},
                    )
                except Exception as e:
                    if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                        # If error suggests using max_completion_tokens instead
                        if self.verbose and False:  # Disable verbose API logging
                            print(f"Retrying with max_completion_tokens parameter")
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=max_tokens,
                        )
                    else:
                        # Re-raise if it's a different error
                        raise
                
                # Print the complete raw response object for debugging
                if self.verbose and False:  # Disable verbose API logging
                    print(f"Raw OpenAI Response: {response}")
                    
                result = response.choices[0].message.content
                
                if self.verbose and False:  # Disable verbose API logging
                    print(f"OpenAI Response Content: {result}")
                
                # Capture token usage
                token_usage = {
                    "question_index": self.question_count,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                
                if is_questioner:
                    self.questioner_token_usage.append(token_usage)
                else:
                    self.answerer_token_usage.append(token_usage)
                
                if self.verbose:
                    print(f"Token usage: {token_usage['total_tokens']} total ({token_usage['prompt_tokens']} prompt, {token_usage['completion_tokens']} completion)")
                
                return result
            
            elif provider == "anthropic":
                # Extract reasoning level from model name if present
                # Format: "claude-3-7-sonnet-20250219-reasoning-{none|low|high}"
                reasoning_level = None
                actual_model = model
                
                if "-reasoning-" in model:
                    parts = model.split("-reasoning-")
                    actual_model = parts[0]
                    reasoning_level = parts[1]
                
                if self.verbose and False:  # Disable verbose API logging
                    print(f"Calling Anthropic API with model: {actual_model}, max_tokens: {max_tokens}")
                    if reasoning_level:
                        print(f"Using reasoning level: {reasoning_level}")
                
                # Set budget_tokens based on reasoning level
                budget_tokens = None
                if reasoning_level == "low":
                    budget_tokens = 2000
                elif reasoning_level == "high":
                    budget_tokens = 40000
                
                # For answerer, always enable reasoning if not explicitly specified
                if not is_questioner and budget_tokens is None:
                    budget_tokens = 2000  # Default to "low" level for answerer
                
                # Build message history for the questioner
                messages = []
                if is_questioner and self.questioner_messages:
                    # Use the stored message history that includes thinking blocks
                    messages = self.questioner_messages.copy()
                    # Append the new user message
                    messages.append({"role": "user", "content": prompt})
                else:
                    # Just use the current prompt if no history or not questioner
                    messages = [{"role": "user", "content": prompt}]
                
                # Build API call parameters
                params = {
                    "model": actual_model,
                    "max_tokens": max_tokens,
                    "messages": messages
                }
                
                # Add thinking with budget_tokens if specified
                if budget_tokens is not None and is_questioner:
                    params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget_tokens
                    }
                
                # Enable reasoning for answerer as well if resoning level is specified
                elif budget_tokens is not None and not is_questioner:
                    # Use a smaller budget for answerer since responses are shorter
                    answerer_budget = min(budget_tokens, 2000)
                    params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": answerer_budget
                    }
                
                # Use streaming for questioner to avoid long-request timeouts
                if is_questioner:
                    response_content = ""
                    thinking_content = None
                    collected_blocks = []
                    
                    with client.messages.stream(**params) as stream:
                        for text in stream.text_stream:
                            response_content += text
                            
                        # Get the final message object
                        response = stream.get_final_message()
                        
                        # Collect all content blocks, including thinking blocks
                        if hasattr(response, 'content') and isinstance(response.content, list):
                            collected_blocks = response.content
                            
                            # Find and log thinking content if available
                            for content_item in collected_blocks:
                                if hasattr(content_item, 'type') and content_item.type == 'thinking':
                                    if hasattr(content_item, 'thinking'):
                                        thinking_content = content_item.thinking
                                        
                                        # Store thinking content for logging
                                        self.questioner_thinking.append({
                                            "question_index": self.question_count,
                                            "thinking": thinking_content
                                        })
                                        
                                        break
                        
                        # Store the message with all blocks for future requests
                        if is_questioner:
                            # Add the user message if not already in history
                            if not self.questioner_messages or self.questioner_messages[-1]["role"] != "user":
                                self.questioner_messages.append({"role": "user", "content": prompt})
                            
                            # Add the assistant response with all content blocks
                            self.questioner_messages.append({
                                "role": "assistant", 
                                "content": collected_blocks
                            })
                        
                        # Use the already collected content
                        result = response_content
                        
                        # Log thinking if verbose mode is enabled
                        if self.verbose and thinking_content:
                            print("\n--- Questioner Thinking ---")
                            print(thinking_content)
                            print("--- End Thinking ---\n")
                        
                        # Get final usage information
                        if hasattr(response, 'usage'):
                            token_usage = {
                                "question_index": self.question_count,
                                "input_tokens": getattr(response.usage, "input_tokens", 0),
                                "output_tokens": getattr(response.usage, "output_tokens", 0),
                                "thinking_tokens": getattr(response.usage, "thinking_tokens", 0),
                                "total_tokens": getattr(response.usage, "total_tokens", 0)
                            }
                            
                            if self.verbose:
                                print(f"Token usage: {token_usage['total_tokens']} total " +
                                    f"({token_usage['input_tokens']} input, " +
                                    f"{token_usage['output_tokens']} output, " +
                                    f"{token_usage['thinking_tokens']} thinking)")
                        
                        # Store token usage data
                        if is_questioner:
                            self.questioner_token_usage.append(token_usage)
                        else:
                            self.answerer_token_usage.append(token_usage)
                else:
                    # For answerer responses, use streaming if reasoning is enabled
                    if "thinking" in params:
                        response_content = ""
                        thinking_content = None
                        collected_blocks = []
                        
                        with client.messages.stream(**params) as stream:
                            for text in stream.text_stream:
                                response_content += text
                                
                            # Get the final message object
                            response = stream.get_final_message()
                            
                            # Collect all content blocks, including thinking blocks
                            if hasattr(response, 'content') and isinstance(response.content, list):
                                collected_blocks = response.content
                                
                                # Find and log thinking content if available
                                for content_item in collected_blocks:
                                    if hasattr(content_item, 'type') and content_item.type == 'thinking':
                                        if hasattr(content_item, 'thinking'):
                                            thinking_content = content_item.thinking
                                            
                                            # Store thinking content for logging
                                            self.answerer_thinking.append({
                                                "question_index": self.question_count,
                                                "thinking": thinking_content
                                            })
                                            
                                            break
                            
                            # Log thinking if verbose mode is enabled
                            if self.verbose and thinking_content:
                                print("\n--- Answerer Thinking ---")
                                print(thinking_content)
                                print("--- End Thinking ---\n")
                                
                            result = response_content
                            
                            # Get usage information
                            if hasattr(response, 'usage'):
                                token_usage = {
                                    "question_index": self.question_count,
                                    "input_tokens": getattr(response.usage, "input_tokens", 0),
                                    "output_tokens": getattr(response.usage, "output_tokens", 0),
                                    "total_tokens": getattr(response.usage, "total_tokens", 0)
                                }
                                
                                if self.verbose:
                                    print(f"Token usage: {token_usage['total_tokens']} total " +
                                        f"({token_usage['input_tokens']} input, {token_usage['output_tokens']} output)")
                            
                            # Store token usage data
                            if is_questioner:
                                self.questioner_token_usage.append(token_usage)
                            else:
                                self.answerer_token_usage.append(token_usage)
                    else:
                        # For short answerer responses without reasoning, use non-streaming
                        response = client.messages.create(**params)
                        
                        # Extract the text content from the response
                        result = ""
                        if hasattr(response, 'content') and len(response.content) > 0:
                            # Find the "text" type content
                            for content_item in response.content:
                                if hasattr(content_item, 'type') and content_item.type == 'text':
                                    result = content_item.text
                                    break
                            
                            # If no "text" type was found but we have content, use the first item's text
                            if not result and hasattr(response.content[0], 'text'):
                                result = response.content[0].text
                        else:
                            # Fallback for accessing response - this may need adjustment based on actual structure
                            print("WARNING: Unexpected response structure. Attempting to extract text.")
                            if hasattr(response, 'text'):
                                result = response.text
                            else:
                                # If we can't determine the structure, print what we got for debugging
                                print(f"Response structure: {dir(response)}")
                                raise Exception(f"Could not extract text from response: {response}")
                
                if self.verbose and False:  # Disable verbose API logging
                    print(f"Anthropic Response Content: {result}")
                return result
            
        except Exception as e:
            print(f"ERROR calling {provider} API: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Error details: {repr(e)}")
            raise
    
    def extract_question(self, response: str) -> Tuple[str, Optional[str]]:
        """
        Extract the question and final guess (if any) from the questioner's response using XML tags.
        
        Args:
            response: Model response
            
        Returns:
            Tuple of (question, final_guess)
        """
        # First check for a guess
        guess_match = re.search(r'<guess>(.*?)</guess>', response, re.IGNORECASE | re.DOTALL)
        guess = None
        if guess_match:
            guess = guess_match.group(1).strip()
        
        # Then extract the question
        question_match = re.search(r'<question>(.*?)</question>', response, re.IGNORECASE | re.DOTALL)
        if question_match:
            question = question_match.group(1).strip()
            return question, guess
        
        # If no question tags found, use the whole response (fallback for backward compatibility)
        if self.verbose:
            print(f"WARNING: Question not properly formatted with XML tags: {response}")
        
        # Check for legacy "My guess is:" format as a fallback
        if guess is None and "My guess is:" in response:
            parts = response.split("My guess is:")
            if len(parts) > 1:
                guess = parts[1].strip()
                response = parts[0].strip()
        
        return response.strip(), guess
    
    def extract_answer(self, response: str) -> Tuple[str, bool, str]:
        """
        Extract the answer from the answerer's response using XML tags only.
        
        Args:
            response: Model response
            
        Returns:
            Tuple of (questioner_answer, entity_guessed flag, full_answer_for_logging)
        """
        # Look for XML tags
        match = re.search(r'<answer>(yes|no|maybe|correct)</answer>', response.lower())
        
        if match:
            answer_type = match.group(1)
            
            # Extract any explanation after the tag
            explanation = response.split('</answer>', 1)
            explanation = explanation[1].strip() if len(explanation) > 1 else ""
            
            # Clean version for questioner (no explanation)
            if answer_type == "correct":
                questioner_answer = "CORRECT!"
                entity_guessed = True
            else:
                questioner_answer = answer_type.capitalize() + "."
                entity_guessed = False
            
            # Full version for logging
            full_answer = f"{questioner_answer}{' ' + explanation if explanation else ''}"
            
            return questioner_answer, entity_guessed, full_answer
        
        # If no XML tags found, return unclear response
        if self.verbose:
            print(f"WARNING: Answer not properly formatted with XML tags: {response}")
        
        return "Unclear.", False, f"Missing proper format: {response}"
    
    def check_success(self, guess: str) -> bool:
        """
        Check if the final guess is correct.
        
        Args:
            guess: Final guess
            
        Returns:
            Whether the guess is correct
        """
        # Clean up the guess
        clean_guess = guess.lower().strip()
        if clean_guess.endswith('.'):
            clean_guess = clean_guess[:-1]
            
        # Clean up the entity
        clean_entity = self.entity.lower().strip()
        
        # Direct match
        if clean_guess == clean_entity:
            return True
        
        # Check if entity is contained in guess or vice versa
        if clean_entity in clean_guess or clean_guess in clean_entity:
            return True
            
        # TODO: Could add more sophisticated matching here, like checking for synonyms
        
        return False
    
    def play_game(self, entity: Optional[str] = None, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Play a full game of 20 Questions.
        
        Args:
            entity: Optional specific entity to use
            category: Optional specific category to use
            
        Returns:
            Game results dictionary
        """
        # Reset game state
        self.questions_asked = []
        self.answers = []
        self.final_guess = None
        self.success = False
        self.question_count = 0
        self.questioner_messages = []  # Reset messages history with thinking blocks
        self.questioner_thinking = []  # Reset thinking content
        self.answerer_thinking = []    # Reset thinking content
        self.questioner_token_usage = []  # Reset token usage tracking
        self.answerer_token_usage = []    # Reset token usage tracking
        
        # Select entity if not provided
        if entity is None:
            self.entity, self.category = self.select_entity(category)
        else:
            self.entity = entity
            self.category = category or "custom"
        
        if self.verbose:
            print(f"\n============= NEW GAME =============")
            print(f"Entity: {self.entity} (Category: {self.category})")
            print(f"Questioner: {self.questioner_provider} - {self.questioner_model}")
            print(f"Answerer: {self.answerer_provider} - {self.answerer_model}")
            print(f"===================================\n")
        
        # Play until max questions or final guess
        extended_game = False
        original_max_questions = self.max_questions
        
        while self.question_count < self.max_questions:
            # Get question from questioner
            questioner_prompt = self.get_questioner_prompt()
            question_response = self.get_model_response(
                self.questioner_provider, 
                self.questioner_model,
                questioner_prompt,
                self.questioner_client,
                is_questioner=True
            )
            
            question, guess = self.extract_question(question_response)
            self.questions_asked.append(question)
            self.question_count += 1
            
            if self.verbose:
                print(f"Q{self.question_count}: {question}")
            
            # Check if there's a final guess
            if guess:
                self.final_guess = guess
                self.success = self.check_success(guess)
                
                if self.verbose:
                    result = "CORRECT! 🎉" if self.success else "INCORRECT ❌"
                    print(f"Final guess: {guess} - {result}")
                    print(f"Actual entity was: {self.entity}")
                    print("===================================\n")
                
                # End the game immediately if the guess is correct
                if self.success:
                    break
                
                # If the guess is wrong, continue asking questions until max_questions is reached
                if not self.success and self.question_count < self.max_questions:
                    if self.verbose:
                        print("Incorrect guess. Continuing with more questions...")
                    self.questions_asked.append(f"(Incorrect guess: {guess})")
                    self.answers.append("That's not correct. Keep asking questions.")
                    continue
                else:
                    break
            
            # Checkpoint after original max questions (usually 20)
            if self.question_count == original_max_questions and not extended_game:
                if self.verbose:
                    print(f"\n===== CHECKPOINT: {original_max_questions} QUESTIONS ASKED =====")
                    print(f"Do you want the questioner to continue asking questions? (y/n)")
                    user_choice = input(">> ").strip().lower()
                    
                    if user_choice == 'y' or user_choice == 'yes':
                        # Extend the game
                        self.max_questions += 30  # Add 10 more questions
                        extended_game = True
                        print(f"Game extended to {self.max_questions} questions total.")
                        
                        # Tell the questioner to keep going
                        continuation_message = f"You've asked {self.question_count} questions, but haven't found the answer yet. Please continue asking questions."
                        self.answers.append(continuation_message)
                        
                        # Skip getting an answer for this question since we're adding our own message
                        continue
                    else:
                        print("Proceeding to final guess...")
            
            # Get answer from answerer
            answerer_prompt = self.get_answerer_prompt()
            answer_response = self.get_model_response(
                self.answerer_provider, 
                self.answerer_model,
                answerer_prompt,
                self.answerer_client,
                is_questioner=False
            )
            
            # Extract both the clean answer for the questioner and full answer for logging
            questioner_answer, entity_guessed, full_answer = self.extract_answer(answer_response)
            
            # Store only the clean answer in the conversation history for the questioner
            self.answers.append(questioner_answer)
            
            if self.verbose:
                # But log the full answer with explanation
                print(f"A: {full_answer}")
                print("---")
            
            # If the answerer indicates the questioner mentioned the correct entity
            if entity_guessed:
                self.success = True
                self.final_guess = self.entity  # Use the actual entity as the guess
                
                if self.verbose:
                    print(f"The questioner mentioned the correct entity in their question!")
                    print(f"Game ended with success after {self.question_count} questions.")
                    print("===================================\n")
                
                break
        
        # If we've reached max questions without a guess, force a guess
        if self.final_guess is None:
            if self.verbose:
                print(f"Reached maximum questions ({self.max_questions}). Forcing final guess...")
                
            force_guess_prompt = (
                f"{self.get_questioner_prompt()}\n\n"
                f"You have used all {self.max_questions} questions. "
                f"You must make your final guess now. What is your best guess for the entity?"
            )
            
            final_response = self.get_model_response(
                self.questioner_provider, 
                self.questioner_model,
                force_guess_prompt,
                self.questioner_client,
                is_questioner=True
            )
            
            # Extract guess from response
            guess = final_response
            if "My guess is:" in final_response:
                guess = final_response.split("My guess is:")[1].strip()
            
            self.final_guess = guess
            self.success = self.check_success(guess)
            
            if self.verbose:
                result = "CORRECT! 🎉" if self.success else "INCORRECT ❌"
                print(f"Final forced guess: {guess} - {result}")
                print(f"Actual entity was: {self.entity}")
                print("===================================\n")
        
        # Compile results
        results = {
            "game_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "questioner_provider": self.questioner_provider,
            "questioner_model": self.questioner_model,
            "answerer_provider": self.answerer_provider,
            "answerer_model": self.answerer_model,
            "entity": self.entity,
            "category": self.category,
            "questions": self.questions_asked,
            "answers": self.answers,
            "question_count": self.question_count,
            "final_guess": self.final_guess,
            "success": self.success,
            "max_questions": self.max_questions,
            "original_max_questions": original_max_questions,
            "extended_game": extended_game,
            "questioner_thinking": self.questioner_thinking,
            "answerer_thinking": self.answerer_thinking,
            "questioner_token_usage": self.questioner_token_usage,
            "answerer_token_usage": self.answerer_token_usage
        }
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """
        Save game results to a JSON file.
        
        Args:
            results: Game results dictionary
        """
        filename = f"{results['game_id']}.json"
        file_path = os.path.join(self.output_dir, filename)
        
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.verbose:
            print(f"Results saved to {file_path}")


class TwentyQuestionsBenchmark:
    """
    Run multiple 20 Questions games to benchmark model performance.
    """
    
    def __init__(
        self,
        output_dir: str = "benchmark_results",
        verbose: bool = True
    ):
        """
        Initialize the benchmark.
        
        Args:
            output_dir: Directory to save results
            verbose: Whether to print progress
        """
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def run_game(
        self,
        entity: str,
        category: str,
        questioner_config: Dict[str, str],
        answerer_provider: str,
        answerer_model: str,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        max_questions: int = 20
    ) -> Dict[str, Any]:
        """
        Run a single game with specified configuration.
        
        Args:
            entity: The entity to use for this game
            category: The category of the entity
            questioner_config: Configuration for the questioner model
            answerer_provider: Provider for the answerer model
            answerer_model: Model name for the answerer
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            max_questions: Maximum number of questions allowed
            
        Returns:
            Game results dictionary
        """
        questioner_provider = questioner_config["provider"]
        questioner_model = questioner_config["model"]
        
        if self.verbose:
            print(f"\nRunning game:")
            print(f"Entity: {entity} (Category: {category})")
            print(f"Questioner: {questioner_provider} - {questioner_model}")
            print(f"Answerer: {answerer_provider} - {answerer_model}")
            print(f"Max Questions: {max_questions}")
        
        game = TwentyQuestionsGame(
            questioner_provider=questioner_provider,
            questioner_model=questioner_model,
            answerer_provider=answerer_provider,
            answerer_model=answerer_model,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            output_dir=os.path.join(self.output_dir, "game_results"),
            verbose=self.verbose,
            max_questions=max_questions
        )
        
        # Play the game with this specific entity
        results = game.play_game(entity=entity, category=category)
        return results
    
    def run_experiment(
        self,
        questioner_models: List[Dict[str, str]],
        answerer_provider: str,
        answerer_model: str,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        num_games_per_model: int = 5,
        categories: Optional[List[str]] = None,
        seed: int = 42,
        max_workers: int = 4,
        max_questions: int = 20
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run an experiment with multiple questioner models against a single answerer model.
        
        Args:
            questioner_models: List of dictionaries with questioner model configs
            answerer_provider: Provider for the answerer model (consistent across all games)
            answerer_model: Model name for the answerer (consistent across all games)
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            num_games_per_model: Number of games to run for each questioner model
            categories: Optional list of categories to use
            seed: Random seed for reproducibility
            max_workers: Maximum number of concurrent workers
            max_questions: Maximum number of questions allowed per game
            
        Returns:
            Tuple of (DataFrame with experiment results, DataFrame with summary)
        """
        random.seed(seed)
        
        if categories is None:
            categories = list(TwentyQuestionsGame.ENTITIES.keys())
        
        # First, prepare a list of entities to use across all models
        test_entities = []
        for category in categories:
            # Get all entities for this category
            category_entities = TwentyQuestionsGame.ENTITIES[category]
            # Calculate how many times we need to cycle through to get num_games_per_model
            num_cycles = (num_games_per_model + len(category_entities) - 1) // len(category_entities)
            # Create a cycling list of entities
            entities_cycle = category_entities * num_cycles
            # Take exactly num_games_per_model entities
            selected_entities = entities_cycle[:num_games_per_model]
            # Add to test entities with category
            for entity in selected_entities:
                test_entities.append((entity, category))
        
        # Generate all game configurations
        all_game_configs = []
        for entity, category in test_entities:
            for questioner_config in questioner_models:
                all_game_configs.append({
                    'entity': entity,
                    'category': category,
                    'questioner_config': questioner_config,
                    'answerer_provider': answerer_provider,
                    'answerer_model': answerer_model,
                    'openai_api_key': openai_api_key,
                    'anthropic_api_key': anthropic_api_key
                })
        
        all_results = []
        
        # Run games in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create futures
            future_to_config = {
                executor.submit(
                    self.run_game,
                    **{**config, 'max_questions': max_questions}  # Pass max_questions to each game
                ): config for config in all_game_configs
            }
            
            # Track progress with tqdm
            total_games = len(all_game_configs)
            completed = 0
            
            if self.verbose:
                print(f"\nRunning {total_games} games in parallel (max {max_workers} concurrent)...")
                progress_bar = tqdm(total=total_games)
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as exc:
                    print(f"Game failed: {config['entity']} with {config['questioner_config']['model']}")
                    print(f"Generated an exception: {exc}")
                
                # Update progress
                completed += 1
                if self.verbose:
                    progress_bar.update(1)
            
            if self.verbose:
                progress_bar.close()
        
        # Compile and save overall results
        df = pd.DataFrame(all_results)
        
        # Calculate summary statistics
        summary = self.calculate_summary(df)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(os.path.join(self.output_dir, f"benchmark_results_{timestamp}.csv"), index=False)
        summary.to_csv(os.path.join(self.output_dir, f"benchmark_summary_{timestamp}.csv"), index=True)
        
        if self.verbose:
            print("\nExperiment complete!")
            print("\nSummary of results:")
            print(summary)
        
        return df, summary
    
    def calculate_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate summary statistics from experiment results.
        
        Args:
            results_df: DataFrame with experiment results
            
        Returns:
            DataFrame with summary statistics
        """
        # Group by model configurations
        grouped = results_df.groupby([
            'questioner_provider', 'questioner_model', 
            'answerer_provider', 'answerer_model', 'category'
        ])
        
        # Extract token usage statistics - this requires additional processing
        # since token_usage is stored as a list in each row
        
        # First, let's create functions to calculate token statistics
        def total_thinking_tokens(row):
            total = 0
            if 'questioner_token_usage' in row and isinstance(row['questioner_token_usage'], list):
                for usage in row['questioner_token_usage']:
                    if isinstance(usage, dict):
                        # Check for thinking_tokens - different API responses may have different structures
                        thinking_tokens = usage.get('thinking_tokens', 0)
                        if thinking_tokens is None:  # Handle None values
                            thinking_tokens = 0
                        total += thinking_tokens
            return total
        
        def total_tokens(row):
            total = 0
            if 'questioner_token_usage' in row and isinstance(row['questioner_token_usage'], list):
                for usage in row['questioner_token_usage']:
                    if isinstance(usage, dict) and 'total_tokens' in usage:
                        total += usage['total_tokens']
            
            if 'answerer_token_usage' in row and isinstance(row['answerer_token_usage'], list):
                for usage in row['answerer_token_usage']:
                    if isinstance(usage, dict) and 'total_tokens' in usage:
                        total += usage['total_tokens']
            return total
        
        # Apply these functions to create new columns
        results_df['total_thinking_tokens'] = results_df.apply(total_thinking_tokens, axis=1)
        results_df['total_tokens'] = results_df.apply(total_tokens, axis=1)
        
        # Include token statistics in the grouped aggregation
        summary = grouped.agg({
            'success': ['mean', 'count'],
            'question_count': ['mean', 'median', 'min', 'max'],
            'total_thinking_tokens': ['mean', 'sum'],
            'total_tokens': ['mean', 'sum']
        })
        
        # Rename columns for clarity
        summary.columns = [
            'success_rate', 'num_games', 
            'avg_questions', 'median_questions', 'min_questions', 'max_questions',
            'avg_thinking_tokens', 'total_thinking_tokens',
            'avg_total_tokens', 'total_tokens'
        ]
        
        # Sort by success rate (descending)
        summary = summary.sort_values('success_rate', ascending=False)
        
        return summary


def main():
    """
    Main function to run the benchmark from command line.
    """
    parser = argparse.ArgumentParser(description="Run 20 Questions benchmark for AI models")
    
    parser.add_argument("--games", type=int, default=5, help="Number of games per model")
    parser.add_argument("--output", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Whether to print progress")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--answerer_provider", type=str, default="anthropic", 
                        help="Provider for answerer model (openai or anthropic)")
    parser.add_argument("--answerer_model", type=str, default="claude-3-7-sonnet-20250219", 
                        help="Model name for answerer")
    parser.add_argument("--max_workers", type=int, default=4, 
                        help="Maximum number of concurrent games to run")
    parser.add_argument("--max_questions", type=int, default=20,
                        help="Maximum number of questions allowed (e.g., 20 for '20 Questions', 30 for '30 Questions')")
    
    args = parser.parse_args()
    
    # Define questioner models to benchmark (using a consistent answerer)
    questioner_models = [

            {
            "provider": "anthropic",
            "model": "claude-3-7-sonnet-20250219-reasoning-high"
        },

    ]
    
    # Run the benchmark
    benchmark = TwentyQuestionsBenchmark(
        output_dir=args.output,
        verbose=args.verbose
    )
    
    results, summary = benchmark.run_experiment(
        questioner_models=questioner_models,
        answerer_provider=args.answerer_provider,
        answerer_model=args.answerer_model,
        openai_api_key=OPENAI_API_KEY,
        anthropic_api_key=ANTHROPIC_API_KEY,
        num_games_per_model=args.games,
        seed=args.seed,
        max_workers=args.max_workers,
        max_questions=args.max_questions
    )
    
    print("\nBenchmark complete!")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()