import dspy

class GenerateOriginalCaptions(dspy.Signature):
    """
    Persona:
    <persona>
    
    Task:
    Generate 30 new diverse and unique upper-body garment captions.
    Ensure the output is maintain consistency with the provided examples.

    Rules:
    - The output should be a json list of 30 captions.
    - Each caption should be a string.
    - The captions should be diverse and unique.
    """
    persona: str = dspy.InputField(
        desc="A detailed description of the persona."
    )
    original_captions: list[str] = dspy.OutputField(
        desc="A list of detailed descriptions of upper-body garment’s visible features."
    )
    
