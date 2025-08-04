from langchain_openai import ChatOpenAI
import json
import logging
import requests
from bs4 import BeautifulSoup
# Optional: Enable logging
logging.basicConfig(level=logging.INFO)

def generate_structured_prompt(json_data):
    try:
        json_preview = json.dumps(json_data[:10], indent=2)  # Safely preview a portion
    except Exception as e:
        logging.error(f"Error processing JSON: {e}")
        json_preview = "{}"

    return f"""
You are an expert Information Architect and Taxonomist. Your task is to deeply analyze the provided website content in JSON format and transform it into a clearly organized, human-readable structure tailored to the specific domain of the website. This transformation should enhance content discoverability, improve navigation, and reflect best practices in modern web taxonomy.

Analyze the content and generate 4 to 5 high-level main categories, each containing 3 to 5 subcategories, followed by a relevant FAQ section. All content should be specific, concise, and adapted to the nature of the website.

1️⃣ CATEGORY IDENTIFICATION  
- Identify the website’s industry type based on its content. Choose from (but not limited to):  
  Manufacturing, Restaurant, Education, Healthcare, E-commerce, Hospitality, SaaS, Non-Profit, Government, Automotive, Real Estate, Fitness, Entertainment, Financial Services, Travel  
- Based on the identified type, define 4 to 5 relevant MAIN CATEGORIES that best represent the website’s structure. These categories should reflect how a real-world website in that industry is typically organized.  
  - Examples:  
    - *Manufacturing*: Products, Capabilities, Industries Served, Compliance  
    - *Restaurant*: Menu, Location, Services, Chef’s Specials  
    - *SaaS*: Features, Pricing, Integrations, Use Cases  

2️⃣ STRUCTURED EXTRACTION  
For each main category:  
- Identify 3 to 5 meaningful subcategories or content segments.  
- Extract only the most relevant and useful information under each subcategory.  
- Do not include filler or marketing fluff — focus on clarity and specificity.  
- Use the following strict format to present the output:

[Main Category Name]  
├─ [Subcategory 1]: [Item 1], [Item 2], [Item 3]  
├─ [Subcategory 2]: [Concise description]  
└─ [Subcategory 3]: [Data point or key info]

3️⃣ FAQ GENERATION  
Create or extract 2 to 4 relevant FAQs based on the content. These should reflect actual user questions a visitor might have. Use this format:

FAQs:  
- Q: [Question related to content, product, service, etc.]  
  A: [Direct, informative answer]  
- Q: [Generate a logical question if missing]  
  A: [Accurate answer inferred from context]

FAQs should cover areas like services, product delivery, pricing, contact, customization, or technical support. If the JSON does not contain questions, generate them logically.

📌 RULES  
- Adapt categories/subcategories to fit the actual data — do not force a rigid structure.  
- Avoid placeholders, markdown symbols, or extraneous formatting.  
- Maintain strict indentation and bullet hierarchy as shown above.  
- Output only the structured content — no markdown headers, explanations, or enclosing tags.  
- Do not include any backticks, markdown code blocks, or JSON formatting (` ```json `).  
- Prioritize clarity, usability, and clean organization.  

🧠 Purpose: The goal is to transform raw, unstructured website content into a logically categorized structure that supports user understanding, content planning, and future integration into knowledge systems or CMS platforms.

---  




JSON Data:
{json_preview}
"""

def new_generate_tags_from_gpt(json_data):
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        prompt = generate_structured_prompt(json_data)
        response = llm.predict(prompt)

        if not response:
            logging.warning("No response returned from model.")
            print("No response returned from model.")
            return {"error": "Empty response from GPT model"}
        
        return {"tags": response}

    except Exception as e:
        logging.exception("Error generating tags from GPT.")
        print("Error generating tags from GPT.")
        return {"error": str(e)}

def scrape_url(url):
    """Function to scrape text content from the URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Example: Extracting title, headings, and first paragraph (you can adjust based on the structure of the webpage)
        title = soup.title.text if soup.title else "No Title"
        headings = [h.text for h in soup.find_all(['h1', 'h2', 'h3'])]
        paragraphs = [p.text for p in soup.find_all('p')]

        content = {
            "url": url,
            "title": title,
            "headings": headings,
            "paragraphs": paragraphs[:5]  # Limiting to the first 5 paragraphs for brevity
        }

        return content

    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return {"url": url, "error": str(e)}

def generate_tags_and_buckets_from_json(urls, json_data):
    # Scrape content from each URL
    scraped_data = [scrape_url(url) for url in urls]

    # Prepare the content for the prompt by formatting it as a preview
    try:
        json_preview = json.dumps(scraped_data[:10], indent=2)  # Preview the first 10 items
    except Exception as e:
        logging.error(f"Error processing JSON: {e}")
        json_preview = "{}"  # In case of error, provide an empty preview

    # Log the preview of the scraped data
    logging.info(f"Preview of scraped data:\n{json_preview}")

    # Construct the Langchain prompt template
    prompt_template = f"""
    I have provided a list of URLs below. Please extract the main content from each webpage and generate relevant tags, categorizing them into appropriate buckets. The tags should describe key topics, products, services, or concepts mentioned on the page, and each tag should be categorized into a relevant bucket. Example buckets could be 'Products', 'Applications', 'Services', 'Industries', 'Solutions', 'Others', etc.

    The output should be in the following JSON format:
    {{
      "Catalogue Name 1": {{
        "Name 1": "Description of the concept, product, service, or industry.",
        "Name 2": "Description of the concept, product, service, or industry."
      }},
      "Catalogue Name 2": {{
        "Name 1": "Description of the concept, product, service, or industry.",
        "Name 2": "Description of the concept, product, service, or industry."
      }}
    }}

    Here is an example format of the JSON output:
    {{
      "Industries": {{
        "Semiconductor": "The semiconductor industry involves the design and fabrication of microchips used in various devices.",
        "Surface Finishing": "Surface finishing refers to processes that improve the appearance, durability, and wear resistance of materials."
      }},
      "Products": {{
        "XYZ Product": "A high-performance product designed to meet the needs of modern manufacturing."
      }},
      "Solutions": {{
        "Cloud-based Solution": "A scalable solution that enables businesses to migrate their operations to the cloud."
      }}
    }}

    Below is the preview of the scraped content from the URLs provided:
    {json_preview}
    """

    logging.info(f"Prompt sent to model: {prompt_template[:500]}")  # Only log a snippet to avoid too much data

    # Initialize the LLM model
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Generate tags and categorize them based on the content from the URLs and the previewed JSON data
    try:
        result = llm.predict(prompt_template)
        logging.info(f"Generated result: {result[:500]}")  # Log a snippet of the result for debugging
        return result
    except Exception as e:
        logging.error(f"An error occurred during the prediction: {e}")
        return None

# # Example: Your scraped data in JSON format
# scraped_data = [
#     {"text": "Inline heaters for industrial applications."},
#     {"text": "Chemical heating solutions for energy efficiency."},
#     {"text": "Innovative thermal systems to improve energy usage."},
#     {"text": "Industrial heating for process heating applications."},
#     {"text": "Solutions for energy savings in manufacturing."}
# ]

# url = 'https://www.processtechnology.com/'




