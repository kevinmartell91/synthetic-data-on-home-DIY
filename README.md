# Synthetic-data-on-home-DIY
Synthetic dataset generation for home repair scenarios using prompt templates across five categories (appliance, plumbing, electrical, HVAC, general). Outputs structured JSON with safety-aware, tool-specific, and step-by-step answers. Includes schema validation, failure mode analysis, and a correction pipeline to improve data quality.


## **Goal**
Regenerate "corrected" versions of failed examples using a second-pass prompt. Re-label and re-analyze if failure rates improve. 

## Tools used 
 - **Braintrust** to log things.
 - **OpenRouter** (unified inference for LLM) to switch across LLM models seamlessly.


## Description of the project
**Generation Phase**: Given 5 prompt templates that ask for home DIY Repair Q&A pairs, generate 20 synthetic QA pairs following this JSON structure:
```json
{
    "question": "...",
    "answer": "...",
    "equiment-problem": "...",
    "tools-required": [
        "..."
    ],
    "steps": [
        "..."
    ],
    "safety_info": "...",
    "tips": "..."
}
 ```


## 05 Prompt Template Categories:
Five types/categories of prompt templates are used to genereate synthetic data for Home DIY Repair Q&A.
    
1. Appliance Repair ( appliance_repair )
 ```code
    Focus: Common household appliances
    Examples: Refrigerators, washing machines, dryers, dishwashers, ovens
    Expert Persona: Expert home appliance repair technician with 20+ years of experience
    Emphasis: Technical details and practical homeowner solutions
```
2. Plumbing Repair ( plumbing_repair )
 ```code
    Focus: Common plumbing issues
    Examples: Leaks, clogs, fixture repairs, pipe problems
    Expert Persona: Professional plumber with extensive residential experience
    Emphasis: Safety for homeowner attempts and realistic solutions
```
3. Electrical Repair ( electrical_repair )
 ```code
    Focus: SAFE homeowner-level electrical work
    Examples: Outlet replacement, switch repair, light fixture installation
    Expert Persona: Licensed electrician specializing in safe home electrical repairs
    Emphasis: Critical safety warnings and when to call professionals
```
4. HVAC Maintenance ( hvac_maintenance )
 ```code
    Focus: Basic HVAC maintenance and troubleshooting
    Examples: Filter changes, thermostat issues, vent cleaning, basic troubleshooting
    Expert Persona: HVAC technician specializing in homeowner maintenance
    Emphasis: Seasonal considerations and maintenance best practices
```
5. General Home Repair ( general_home_repair )
 ```code
    Focus: Common general repairs and maintenance
    Examples: Drywall repair, door/window problems, flooring issues, basic carpentry
    Expert Persona: Skilled handyperson with general home repair expertise
    Emphasis: Material specifications and practical DIY solutions
```
## Template Selection Strategy:
- **Random Selection**: Each generated sample randomly chooses one of the 5 templates for **`diversity`**.

- **Balanced Coverage**: Over 20 samples, this ensures good coverage across all repair categories.

- **Consistent Structure**: All templates produce the same `JSON structure` with 7 required fields:
    ```json
    {
        "question": "...",
        "answer": "...",
        "equiment-problem": "...",
        "tools-required": [
            "..."
        ],
        "steps": [
            "..."
        ],
        "safety_info": "...",
        "tips": "..."
    }
    ```

## Template Design Principles:
- **Domain Expertise**: Each template uses a specific expert persona.
- **Safety Focus**: Strong emphasis on safety warnings and when to call professionals.
- **Practical Scope**: Limited to repairs that are safe and practical for homeowners.
- **Structured Output**: Consistent JSON format for downstream processing.
- **Realistic Scenarios**: Focus on common, real-world repair situations.

## Important considerations
> ### **Diversity**
> This diverse template approach ensures the synthetic data covers the full spectrum of home DIY repair scenarios while maintaining consistency in structure and quality.

> ### **Validation Phase**
> The use Pydantic or jsonschema validates that outputs are structurally correct. Filter invalid entries before moving to error analysis. 

> ### **Failure Labeling**
> **Pandas DataFrame created**:
> - with Trace ID (`auto-assigned`)
> - all structured fields (e.g., answer, ...), and
> - binary columns for each of the 6 failure modes ( `0 = success`, `1 = failure`)  

> ### **Manually labeled 20 entries / use LLM auto-label common failures**. 
> FAILURE MODES: 
>  - Incomplete Answer (`incomplete_answer`)
>  - Safety Violations (`safety_violations`)
>  - Unrealistic Tools (`unrealistic_tools`)
>  - Overcomplicated Solution (`overcomplicated_solution`)
>  - Missing Context (`missing_context`)
>  - Poor Quality Tips (`poor_quality_tips`).

> ### **Analysis**
> - **Heatmap of failure modes**: used across samples created, which identifies the most common failure types, and
> - **Correlations**: (e.g., `Overcomplicated Recipes` ↔ `Missing Equipment`)  


