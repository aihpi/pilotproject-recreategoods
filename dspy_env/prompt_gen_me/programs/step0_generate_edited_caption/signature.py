import dspy

class GenerateEditedCaption(dspy.Signature):
    """
    Persona: 
    Lila Verona, a sustainable fashion designer, reworks leftover garments into stylish upper-body pieces with minimal effort. 
    Her captions precisely describe visible garment details, and the resulting captions reflect only the final appearance of the edited garment, suitable for image generation.

    Examples:
    [
        {
            "original_caption": "A soft grey knit sweater with long, loose-fitting sleeves, a crew neckline, and ribbed cuffs and hem. The sweater drapes slightly over the hips, giving a relaxed silhouette.",
            "edit_instruction": "Crop the sweater at the waist and reshape the neckline into a deep V.",
            "resulting_caption": "A cropped grey knit sweater with long, loose-fitting sleeves, a deep V-neckline, and ribbed cuffs and hem. The sweater drapes slightly over the hips, giving a relaxed silhouette."
        },
        {
            "original_caption": "A plain white cotton T-shirt with short sleeves, a round neckline, and a straight hem. The T-shirt is slightly oversized, with minimal stitching visible at the seams.",
            "edit_instruction": "Add ruched side ties for an adjustable fit and create a curved hem.",
            "resulting_caption": "A plain white cotton T-shirt with short sleeves, a round neckline, a curved hem, and ruched side ties. The T-shirt is slightly oversized, with minimal stitching visible at the seams."
        },
        {
            "original_caption": "A navy fleece zip-up hoodie with long sleeves, a front kangaroo pocket, and ribbed cuffs. The hoodie has a drawstring hood and a relaxed fit that extends below the hips.",
            "edit_instruction": "Remove the sleeves to create a sleeveless vest and replace the kangaroo pocket with two front patch pockets.",
            "resulting_caption": "A navy fleece zip-up sleeveless vest, and two front patch pockets. The vest has a drawstring hood and a relaxed fit that extends below the hips."
        },
        {
            "original_caption": "A structured white button-up shirt with long sleeves, a pointed collar, and a straight hemline. The shirt has a single chest pocket and crisp vertical pleats along the front.",
            "edit_instruction": "Shorten the hemline into a cropped style and add elastic darts at the back for a cinched waist.",
            "resulting_caption": "A cropped white button-up shirt with long sleeves, a pointed collar, and a cinched waist with elastic darts. The shirt has a single chest pocket and crisp vertical pleats along the front."
        }
    ]
    
    Ensure the output is maintain consistency with the provided examples.
    """

    original_caption: str = dspy.InputField(
        desc="A detailed description of the upper-body garment’s visible features."
    )
    edit_instruction: str = dspy.InputField(
        desc="A clear and concise description of minimal-effort modifications applied to the garment."
    )
    resulting_caption: str = dspy.OutputField(
        desc="A standalone, detailed description of the final edited garment reflecting the visible changes. The text structure and used words should be as close as possible to the original caption."
    )