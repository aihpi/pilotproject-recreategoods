import dspy

class GenerateEditInstructions(dspy.Signature):
    """
    Persona: 
    Lila Verona, a sustainable fashion designer, reworks leftover garments into stylish upper-body pieces with minimal effort.
    
    Examples:
    "original_caption": "A soft grey knit sweater with long, loose-fitting sleeves, a crew neckline, and ribbed cuffs and hem. The sweater drapes slightly over the hips, giving a relaxed silhouette.",
    "edit_instruction": "Crop the sweater at the waist and reshape the neckline into a deep V.",
    
    "original_caption": "A plain white cotton T-shirt with short sleeves, a round neckline, and a straight hem. The T-shirt is slightly oversized, with minimal stitching visible at the seams.",
    "edit_instruction": "Add ruched side ties for an adjustable fit and create a curved hem.",

    "original_caption": "A navy fleece zip-up hoodie with long sleeves, a front kangaroo pocket, and ribbed cuffs. The hoodie has a drawstring hood and a relaxed fit that extends below the hips.",
    "edit_instruction": "Remove the sleeves to create a sleeveless vest and replace the kangaroo pocket with two front patch pockets.",
    
    "original_caption": "A structured white button-up shirt with long sleeves, a pointed collar, and a straight hemline. The shirt has a single chest pocket and crisp vertical pleats along the front.",
    "edit_instruction": "Shorten the hemline into a cropped style and add elastic darts at the back for a cinched waist.",

    Task:
    Generate 3 different edit instructions for the given original caption which transforms the garment into a different style.
    The edit instructions should be minimal changes.
    The edit instructions should be specific and clear.
    The edit instructions should omit any mention of the edit's purpose.
    Ensure the output is maintain consistency with the provided examples.

    Rules:
    - The output should be a json list of 3 edit instructions.
    - Each edit instruction should be a string.
    - Each edit instruction should be a minimal change.
    - Each edit instruction could consist of multiple small changes.
    """
    original_caption: str = dspy.InputField(
        desc="A detailed description of upper-body garment’s visible features."
    )
    edit_instructions: list[str] = dspy.OutputField(
        desc="A list of clear and concise descriptions of minimal-effort modifications applied to the garment."
    )
    