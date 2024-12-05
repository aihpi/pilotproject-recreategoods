from sentence_transformers import SentenceTransformer, util

# Load pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Input data
input_caption = "A black leather jacket with long sleeves"
edit_instructions = "Remove the sleeves"
output_caption = "A black leather jacket without sleeves"

# Generate embeddings
input_embedding = model.encode(input_caption, convert_to_tensor=True)
edit_embedding = model.encode(edit_instructions, convert_to_tensor=True)
output_embedding = model.encode(output_caption, convert_to_tensor=True)

# Combine input and edit embeddings
combined_embedding = input_embedding + edit_embedding

# Compute cosine similarities
similarity_combined_output = util.cos_sim(combined_embedding, output_embedding).item()
similarity_input_output = util.cos_sim(input_embedding, output_embedding).item()
similarity_edit_output = util.cos_sim(edit_embedding, output_embedding).item()

# Display results
print(f"Similarity between Combined (Input + Edit) and Output: {similarity_combined_output:.4f}")
print(f"Similarity between Input and Output: {similarity_input_output:.4f}")
print(f"Similarity between Edit Instructions and Output: {similarity_edit_output:.4f}")

# Interpretation
if similarity_combined_output > max(similarity_input_output, similarity_edit_output):
    print("The output aligns well with both the input and the edit instructions.")
else:
    print("The output may not fully reflect the input and/or edit instructions.")