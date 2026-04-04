"""
Resolution mode prompts for DIY repair QA generation
"""

resolution_mode_prompts = {
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
