import json

# Given JSON structure
with open('predictions3.json', 'r') as input_file:
    input_json = json.load(input_file)

# Process each entry in the input JSON
output_json = {
    "info": {},
    "images": [],
    "annotations": []
}

# Counter variables for image and annotation IDs
image_id_counter = 1
annotation_id_counter = 1

for input_entry in input_json:
    # Desired JSON structure for each entry
    output_entry = {
        "id": image_id_counter,
        "file_name": f"./images/{input_entry['image_name']}",
    }

    # Extracting information from the given JSON entry
    image_name = input_entry["image_name"]
    coordinates_str = image_name.split("[")[1].split("]")[0]
    coordinates = [int(coord) for coord in coordinates_str.split(",")]

    # Filling in the desired JSON structure with extracted information

    output_json["images"].append(output_entry)

    # Annotation for each image
    annotation_entry = {
        "id": annotation_id_counter,
        "category_id": 1,
        "image_id": image_id_counter,
        "area": (coordinates[2] - coordinates[0]) * (coordinates[3] - coordinates[1]),
        "bbox": [
            input_entry["predicted_coordinates"][0],
            input_entry["predicted_coordinates"][1],
            16,
            16
        ]
    }

    output_json["annotations"].append(annotation_entry)

    # Increment counters
    image_id_counter += 1
    annotation_id_counter += 1

# Save the resulting JSON to an output file
with open('output3.json', 'w') as output_file:
    json.dump(output_json, output_file, indent=2)
