import xml.etree.ElementTree as ET

def load_xmls(xml_name):
    # Read the XML file
    tree = ET.parse(xml_name)
    root = tree.getroot()

    # Get the total number of reviews
    review_count = len(root.findall('Review'))

    # Calculate the number of reviews per part
    reviews_per_part = 35
    num_parts = (review_count + reviews_per_part - 1) // reviews_per_part

    # Split the reviews into parts
    reviews = root.findall('Review')
    parts = [reviews[i:i+reviews_per_part] for i in range(0, review_count, reviews_per_part)]

    # Save each part to a separate file
    for i, part in enumerate(parts):
        part_root = ET.Element('Reviews')
        part_root.extend(part)
        part_tree = ET.ElementTree(part_root)
        part_tree.write(f'part{i+1}.xml')

    print(f'{num_parts} parts created.')
