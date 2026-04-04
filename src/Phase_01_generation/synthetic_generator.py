"""
Synthetic gnerator for DIY repair QA for repair resolution
This module generates synthetic QA pair for LLM-as-judge evaluation.
"""

import random, json
from datetime import datetime
from typing import Dict, List, Any, Type, TypeVar, Optional, Tuple, Union
from pydantic_classes import (
    DIYRepairSyntheticItem,
    OutputStructureBase,
    Metadata,
    MalformedOuputStructure,
    DIYRepairSyntheticMalformedItem,
)
from pydantic import BaseModel, ValidationError
from pipeline_core.llm import chat as llm_chat
from pipeline_core.utils import try_parse_json
from ._issue_types import *
from ._templates import *
from ._ai_generated_queries import *

T = TypeVar("T", bound=BaseModel)


class SyntheticGenerator:
    """Generates a Synthetic DIY repair QA pairs - Synthtic generatetor class"""

    def __init__(self):

        self.phase_name = "01_generation"

        self.max_attemps = 1  # retries on malformed JSON or schema errors

        self.issue_types = issue_types

        self.templates = templates

        self.ai_generated_queries = ai_generated_queries

    def generate_query(self, issue_type: str) -> str:

        try:
            random_choice = random.choice(self.ai_generated_queries[issue_type])
            return random_choice

        except KeyError:
            raise ValueError(f"Invalid issue type: {issue_type}")

    def generate_prompt(self, issue_type: str, user_query: str, schema=Type[T]) -> str:
        """Generates a prompt for the LLM to generate a response with a given schema"""

        prompt = self.templates.get(issue_type)
        prompt += f"\nISSUE TYPE: {issue_type}"
        prompt += f"\n\nUSER QUERY: {user_query}"
        prompt += "\n\nKeep response concise (2-4 sentences)."
        prompt += (
            "\n\nRespond in JSON format according to the following Pydantic schema:\n"
        )
        prompt += "\n" + schema.model_json_schema().__str__()

        return prompt

    def generate_structured(
        self,
        id: str,
        user_query: str,
        issue_type: str,
        params: Dict[str, Any],  # openrouter params for chat.completion
        schema: Type[T],
    ) -> Tuple[Optional[T] | None, Optional[MalformedOuputStructure] | None]:
        """
        Generates different types of DIY responses based on user query and issue type

        Call LLM → parse JSON → validate with Pydantic.
        Retries on malformed JSON or schema errors.
        Returns None on hard failure (caller decides what to do).

        """

        #  llm prompt
        prompt = self.generate_prompt(
            issue_type=issue_type, user_query=user_query, schema=schema
        )

        # braintrust metadata for logging
        metadata = {
            "phase": self.phase_name,
            "phase_args": {
                "id": id,
                "user_query": user_query,
                "issue_type": issue_type,
            },
            "phase_tags": [
                self.phase_name,
                f"issue_{issue_type}",
            ],
        }

        last_error = None

        for attempt in range(1, self.max_attemps + 1):
            try:
                raw = llm_chat(
                    prompt=prompt,  # llm prompt
                    params=params,  # openrouter params for chat.completion
                    metadata=metadata,  # braintrust metadata for logging
                )

                ok, data = try_parse_json(raw)
                if not ok:
                    last_error = MalformedOuputStructure(
                        error_message=f"[{self.phase_name}] JSON parse failed on attempt {attempt} for id: {id}",
                        malformed_json=raw,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    continue

                try:
                    print(
                        "[{self.phase_name}] Schema validation success for id: {id}",
                        schema.model_validate(data),
                    )
                    return (schema.model_validate(data), None)

                except ValidationError as e:
                    last_error = MalformedOuputStructure(
                        error_message=f"[{self.phase_name}] Schema validation failed in attempt {attempt}: [ERROR] - {e}",
                        malformed_json=raw,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    continue

            except Exception as e:
                last_error = MalformedOuputStructure(
                    error_message=f"[{self.phase_name}] Error generating structured response: [ERROR] - {e}",
                    malformed_json=raw,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                print(last_error)
                continue

        # here all attempts failed
        print(
            f"[{self.phase_name}] Failed to generate structured response for id: {id} after {self.max_attemps} attempts. Last error: {last_error}"
        )

        return (None, last_error)

    def generate_synthetic_dataset(
        self, num_samples: int = 20
    ) -> List[Union[DIYRepairSyntheticItem, DIYRepairSyntheticMalformedItem]]:
        """Generates a dataset of synthetic DIY repair QA pairs with especific number of samples"""

        valid_samples = []
        failed_samples = []

        for idx, _ in enumerate(range(num_samples)):
            item_id = f"qa_{str(idx+1).zfill(3)}"

            # randomly select a category or issue type
            issue_type = random.choice(self.issue_types)

            # given category, generate user query
            user_query = self.generate_query(issue_type)

            params = {
                "max_tokens": 500,  # Increased max_tokens
                "temperature": 0.7,  # Adjusted temperature slightly
            }

            # generate llm response and log  with braintrust
            item, error = self.generate_structured(
                id=item_id,
                user_query=user_query,
                issue_type=issue_type,
                params=params,
                schema=OutputStructureBase,
            )

            if item is not None:
                item = DIYRepairSyntheticItem(
                    id=item_id,
                    **item.model_dump(),
                    metadata=Metadata(issue_type=issue_type, user_query=user_query),
                    error=None,
                )
                valid_samples.append(item)

            if error:
                # log failed sample for debugging
                failed_item = DIYRepairSyntheticMalformedItem(
                    id=item_id,
                    metadata=Metadata(issue_type=issue_type, user_query=user_query),
                    error=error,
                )
                failed_samples.append(failed_item)
                continue

        # return both successful and failed samples to be processed in Phase 02 Structural Validation
        return valid_samples + failed_samples
