"""
Synthetic gnerator for DIY repair QA for repair resolution
This module generates synthetic QA pair for LLM-as-judge evaluation.
"""

import random, json
from typing import Dict, List, Any, Type, TypeVar, Optional, Tuple
from pydantic_classes import DIYRepairSyntheticItem, OutputStructureBase
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
    ) -> Optional[T] | None:
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
                    last_error = f"[{self.phase_name}] JSON parse failed on attempt {attempt} for id: {id}"
                    continue

                try:
                    print(
                        "[{self.phase_name}] Schema validation success for id: {id}",
                        schema.model_validate(data),
                    )

                    return schema.model_validate(data)

                except ValidationError as e:
                    last_error = f"[{self.phase_name}] Schema validation failed in attempt {attempt}: {e}"
                    continue

            except Exception as e:
                print(
                    f"[{self.phase_name}] Error generating structured response: [ERROR] - {e}"
                )
                continue

        # here all attempts failed
        print(
            f"[{self.phase_name}] Failed to generate structured response for id: {id} after {self.max_attemps} attempts. Last error: {last_error}"
        )
        return None

    def generate_synthetic_dataset(
        self, num_samples: int = 20
    ) -> List[DIYRepairSyntheticItem]:
        """Generates a dataset of synthetic DIY repair QA pairs with especific number of samples"""

        samples = []
        failed_sample_items = []

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
            item = self.generate_structured(
                id=item_id,
                user_query=user_query,
                issue_type=issue_type,
                params=params,
                schema=OutputStructureBase,
            )

            if item is not None:
                item = DIYRepairSyntheticItem(id=item_id, **item.model_dump())
                samples.append(item)

            else:
                # log failed sample for debugging
                failed_sample_items.append(
                    {
                        "id": item_id,
                        "user_query": user_query,
                        "issue_type": issue_type,
                    }
                )
                continue

            # TODO: handle failed sample items

        return samples
