
import requests
import json

# llama3.2:1b, llama3.1:8b, qwen2.5:14b
def send_prompt_to_ollama(prompt, model="qwen2.5:14b"):
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        response_text = ""
        for line in response.text.splitlines():
            if line.strip():
                result = json.loads(line)
                if result.get('response'):
                    response_text += result['response']
                if result.get('done', False):
                    break
        return response_text
    except requests.exceptions.RequestException as e:
        print(f"Error making request to Ollama: {e}")
        return None
    
prompt = """
Persona: 
Lila Verona, a sustainable fashion designer, reworks leftover garments into stylish upper-body pieces with minimal effort. Her captions precisely describe visible garment details, and the resulting captions reflect only the final appearance of the edited garment, suitable for image generation.

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

Task Directive:

“Generate 100 examples in the following JSON format. Each example should include:
	1.	original_caption: A detailed description of the upper-body garment’s visible features.
	2.	edit_instruction: A clear and concise description of minimal-effort modifications applied to the garment.
	3.	resulting_caption: A standalone, detailed description of the final edited garment reflecting only the visible changes.

Ensure the output is in valid JSON format, and maintain consistency with the provided examples.”

"""
result = send_prompt_to_ollama(prompt)
print(result)

# part of the llama3.1:8b response
#
# # Convert to dataframe
# df = pd.DataFrame(data)

# # Drop duplicates
# df.drop_duplicates(inplace=True)

# # Group by original caption and edit instruction
# grouped_df = df.groupby(['original_caption', 'edit_instruction'])

# # Apply a function to each group
# def process_group(group):
#     # Get the unique values of each column
#     unique_original_caption = group['original_caption'].unique()[0]
#     unique_edit_instruction = group['edit_instruction'].unique()[0]

#     # Return a dictionary with the original caption and edit instruction
#     return {'original_caption': unique_original_caption, 'edit_instruction': unique_edit_instruction}

# # Apply the function to each group
# processed_groups = grouped_df.apply(process_group)

# # Print the result
# print(processed_groups)

#######
# # Define a function to create an object
# def create_object(obj):
#     # Create a dictionary from the input string
#     obj_dict = {'original': obj['original_caption'], 
#                 'edit_instruction': obj['edit_instruction'],
#                 'resulting': obj['resulting_caption']}
    
#     return obj_dict

# # Apply the function to each element in the list
# new_list = [create_object(obj) for obj in garment_list]

# print(new_list)

# # Define a function to calculate the difference between two lists
# def list_difference(list1, list2):
#     # Use set differences to find elements that are in one list but not the other
#     diff_set = set(list1) - set(list2)
    
#     # Convert the set back into a list and return it
#     return list(diff_set)

# # Define a function to calculate the similarity between two lists
# def list_similarity(list1, list2):
#     # Use set intersections to find elements that are in both lists
#     sim_set = set(list1) & set(list2)
    
#     # Convert the set back into a list and return it
#     return list(sim_set)

# # Calculate the difference between the two lists
# diff_list = list_difference(new_list, new_list[:5])

# print(diff_list)

# # Calculate the similarity between the two lists
# sim_list = list_similarity(new_list, new_list[:5])

# print(sim_list)
