"""
Synthetic gnerator for DIY repair QA for repair resolution
This module generates synthetic QA pair for LLM-as-judge evaluation.
"""

import random, os

# Try to load from .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed, that's fine

from braintrust import current_span, init_logger, start_span, traced
from openai import OpenAI
from pydantic import BaseModel, Field, parse_obj_as
from typing import Dict, List, Any

# Initialize Braintrust logger
logger = init_logger(
    project="mini-project-01",
    api_key=os.getenv("BRAINTRUST_API_KEY") or "NOT SET",
)


# declare the OutputStructure pydantic class
class OutputStructure(BaseModel):
    question: str = Field(description="The user's query")
    answer: str = Field(description="Sythetic answer generated")
    equipment_problem: str = Field(
        description="Equipment required to solve the problem"
    )
    tools_required: str = Field(description="Tools required to solve the problem")
    step: str = Field(description="Step by step instructions")
    safety_info: str = Field(description=" Safety information is available")
    tips: str = Field(description="Tips to solve the problem")


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY") or "your_openrouter_api_key_here",
)


class SyntheticGenerator:
    """Generate a Synthtic generatetor class"""

    def __init__(self):

        self.issue_types = [
            "appliance_repair",
            "plumbing_repair",
            "electrical_repair",
            "hvac_repair",
            "general_home_repair",
        ]

        self.templates = {
            "appliance_repair": f"""
                You are an EXPERT home appliance repair technician with 20+ years of experience who
                FOCUCES in Common household appliances (Refrigerators, washing machines, dryers, dishwashers, ovens), and
                response with EMPHASIS ON Technical details and practical homeowner solutions.
            """,
            "plumbing_repair": f"""
                You are an EXPERT Professional plumber with extensive residential experience who
                FOCUSES in Common plumbing issues such as: Leaks, clogs, fixture repairs, pipe problems, and
                reponse with EMPHASIS in Safety for homeowner attempts and realistic solutions

            """,
            "electrical_repair": f"""
                You are an EXPERT Licensed electrician specializing in safe home electrical repairs who
                FOCUSES in SAFE homeowner-level electrical work such as: Outlet replacement, switch repair, light fixture installation, and
                reponse with EMPHASIS in Critical safety warnings and when to call professionals

            """,
            "hvac_repair": f"""
                You are an EXPERT HVAC technician specializing in homeowner maintenance who
                FOCUSES in Basic HVAC maintenance and troubleshooting such as: Filter changes, thermostat issues, vent cleaning, basic troubleshooting, and
                reponse with EMPHASIS in Seasonal considerations and maintenance best practices

            """,
            "general_home_repair": f"""
                You are an EXPERT Skilled handyperson with general home repair expertise who
                FOCUSES in Common general repairs and maintenance such as: Drywall repair, door/window problems, flooring issues, basic carpentry, and
                reponse with EMPHASIS in Material specifications and practical DIY solutions

            """,
        }

        self.resolution_mode_prompts = {
            "complete_solution": f"""
                Generate a DIY repair support bot response that provides a COMPLETE SOLUTION:

                The response should:
                - Offer clear, step-by-step instructions tailored to the repair issue
                - Include necessary tools, safety precautions, and materials
                - Be actionable, thorough, and easy to follow for a DIY user
                - Use a professional and encouraging tone
            """,
            "incomplete_answer": f"""
                Generate a DIY repair response that provides an INCOMPLETE ANSWER, requiring further interaction:

                The response should:
                - Address only part of the repair issue
                - Leave out critical steps or materials needed
                - Cause confusion or require the user to guess what to do next
            """,
            "safety_violations": f"""
                Generate a DIY repair response that includes SAFETY VIOLATIONS:

                The response should:
                - Recommend actions that could be hazardous or unsafe for a homeowner
                - Ignore standard safety precautions like turning off power or water
                - Be inappropriate for non-professional users
            """,
            "unrealistic_tools": f"""
                Generate a DIY repair response that suggests UNREALISTIC TOOLS or resources:

                The response should:
                - Recommend tools or equipment that are impractical for typical DIY users
                - Assume access to professional-grade or expensive items
                - Be unrealistic for a home repair scenario
            """,
            "overcomplicated_solution": f"""
                Generate a DIY repair response that provides an OVERCOMPLICATED SOLUTION:

                The response should:
                - Include unnecessary steps or overly technical language
                - Make the repair harder than it needs to be
                - Confuse the user with complex or irrelevant instructions
            """,
            "missing_context": f"""
                Generate a DIY repair response that suffers from MISSING CONTEXT:

                The response should:
                - Fail to consider key details from the user's repair situation
                - Make assumptions that don’t fit the described problem
                - Provide advice that doesn’t apply due to lack of understanding
            """,
            "poor_quality_tips": f"""
                Generate a DIY repair response that includes POOR QUALITY TIPS:

                The response should:
                - Offer vague, unreliable, or untested advice
                - Lack clarity or usefulness
                - Be unlikely to help the user resolve their repair issue effectively
            """,
        }

    def generate_query(self, issue_type: str) -> str:
        """AI-generated queries for issue types"""
        queries = {
            "appliance_repair": [
                "My refrigerator is making a loud humming noise and not cooling properly.",
                "The washing machine won’t drain and leaves clothes soaking wet.",
                "My dryer spins but doesn’t produce any heat.",
                "The dishwasher isn’t cleaning dishes well and leaves residue.",
                "The oven takes too long to preheat and sometimes shuts off mid-cycle.",
            ],
            "plumbing_repair": [
                "There’s a constant drip from my bathroom faucet even when it’s turned off.",
                "My kitchen sink is clogged and water backs up every time I run it.",
                "The toilet keeps running after flushing and won’t stop unless I jiggle the handle.",
                "I found a small leak under the bathroom sink where the pipe connects to the wall.",
                "The shower pressure dropped suddenly and barely sprays water now.",
            ],
            "electrical_repair": [
                "I replaced an outlet but now it doesn’t seem to have any power.",
                "The light switch feels loose and sometimes doesn’t turn the light on.",
                "I installed a new ceiling light fixture but it flickers constantly.",
                "One of my outlets sparks when I plug something in.",
                "I’m trying to replace a dimmer switch but the wiring looks confusing.",
            ],
            "hvac_repair": [
                "My thermostat won’t turn on even after replacing the batteries.",
                "The air filter looks dirty and I’m not sure how to replace it.",
                "There’s barely any airflow coming from the vents in one room.",
                "The AC unit outside is making a rattling noise and vibrating.",
                "I set the heat to 70 but the house stays cold and the system runs constantly.",
            ],
            "general_home_repair": [
                "There’s a hole in the drywall from a doorknob that I need to patch.",
                "My front door sticks and won’t close smoothly anymore.",
                "The window won’t stay open—it keeps sliding down on its own.",
                "One of the floorboards in the hallway is loose and creaks loudly.",
                "The cabinet door fell off because the hinge screws pulled out of the wood.",
            ],
        }
        try:
            random_choice = random.choice(queries[issue_type])
            return random_choice
        except KeyError:
            raise ValueError(f"Invalid issue type: {issue_type}")

    # notrace_io=True prevents logging the function arguments automatically,
    # so we can log structured input/output ourselves.
    @traced(type="llm", name="generate_varied_response", notrace_io=True)
    def generate_varied_response(
        self, user_query: str, issue_type: str, response_type: str
    ):
        """Generates different types of DIY responses based on user query and issue type"""

        prompt = self.templates.get(issue_type)
        prompt += f"\n\nUSER QUERY: {user_query}"
        prompt += (
            f"\n\nRESPONSE INSTRUCTIONS: {self.resolution_mode_prompts[response_type]}"
        )
        prompt += "\n\nKeep response concise (2-4 sentences)."
        prompt += (
            "\n\nRespond in JSON format according to the following Pydantic schema:\n"
        )
        prompt += OutputStructure.model_json_schema().__str__()

        try:

            params = {
                "max_tokens": 500,  # Increased max_tokens
                "temperature": 0.7,  # Adjusted temperature slightly
            }
            response = client.chat.completions.create(
                model="openai/gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
            )
            # Parse the response text into the Pydantic model
            response_text = (
                response.choices[0].message.content if response.choices else ""
            )
            usage = response.usage

            #  OpenRouter LLM integration
            if logger is not None:
                try:
                    current_span().log(
                        input=[{"role": "user", "content": prompt}],
                        output=response_text,
                        metrics={
                            "prompt_tokens": usage.prompt_tokens if usage else 0,
                            "completion_tokens": (
                                usage.completion_tokens if usage else 0
                            ),
                            "tokens": usage.total_tokens if usage else 0,
                        },
                        metadata={
                            "user_query": user_query,
                            "issue_type": issue_type,
                            "response_type": response_type,
                            **params,
                        },
                    )
                except Exception as e:
                    print(f"Warning: Failed to log to Braintrust: {e}")

            return response_text

        except Exception as e:
            print(f"Error generating response: [ERROR] - {e}")
            return "I appologize, but I'm having trouble processing your request right now."

    def generate_dataset(self, num_samples: int = 20) -> List[Dict[str, Any]]:
        """Generates a dataset of synthetic DIY repair QA pairs with especific number of samples"""
        dataset = []
        for _ in range(num_samples):
            issue_type = random.choice(self.issue_types)
            response_type = random.choice(list(self.resolution_mode_prompts))
            user_query = self.generate_query(issue_type)

            sample = self.generate_varied_response(
                user_query, issue_type, response_type
            )

            # Ensure the sample is a dictionary before appending
            if isinstance(sample, OutputStructure):
                dataset.append(sample.model_dump())
            elif isinstance(sample, dict):
                dataset.append(sample)
            else:
                print(f"Skipping invalid sample type: {type(sample)}")

        return dataset


def main():

    generator = SyntheticGenerator()
    dataset = generator.generate_dataset(num_samples=3)
    print(dataset)


if __name__ == "__main__":
    main()
