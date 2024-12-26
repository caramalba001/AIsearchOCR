import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from pythaiid import thaiid
import json
import re
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

OCR_ENDPOINT = 'YOUR_OCR_ENDPOINT'
OCR_KEY = 'YOUR_OCR_KEY'
AI_API_KEY = "YOUR_AI_API_KEY"
AI_API_VERSION = "2024-02-01"
AI_ENDPOINT = "YOUR_AI_ENDPOINT"

client_OCR = ImageAnalysisClient(endpoint=OCR_ENDPOINT, credential=AzureKeyCredential(OCR_KEY))
client_AI = AzureOpenAI(api_key=AI_API_KEY, api_version=AI_API_VERSION, azure_endpoint=AI_ENDPOINT)

def scrape_google(query, num_results=5):
    query = query.replace(' ', '+')
    url = f'https://www.google.com/search?q="{query}"&num={num_results}'

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Check if "No results found" message exists
        no_results_message = any("ไม่พบผลการค้นหาสำหรับ" in div.get_text() for div in soup.find_all('div'))
        if no_results_message:
            return []  # Return an empty list if no results are found

        results = []
        result_divs = soup.find_all('div', class_='tF2Cxc')
        for div in result_divs:
            title = div.find('h3').get_text(strip=True) if div.find('h3') else "No Title"
            link = div.find('a')['href'] if div.find('a') else "No Link"
            description = div.find('div', class_='VwiC3b').get_text(strip=True) if div.find('div', class_='VwiC3b') else "No Description"
            results.append({
                "Title": title,
                "Link": link,
                "Description": description
            })        
        
        print(results, "\n")

        return results
    else:
        return None

def query_with_AI(query_thai, query_eng, model='gpt-4o'):
    search_result_thai = scrape_google(query_thai)
    search_result_eng = scrape_google(query_eng)

    prompt = f"""
    Extract the person's occupation, education, hobbies, social media profiles, and other relevant personal details from the following information. Provide the data in a structured JSON format suitable for marketing use. Use the person's Thai name {query_thai} and English name {query_eng}.
    
    You must extract the Entities below:\n
    - Current_Occupation: The person's current occupation.\n
    - Previous_Occupation: All previous occupations the person has held.\n
    - Background: The person's background based on their previous occupations.\n
    - Education: All educational details about the person.\n
    - Lifestyle: Guess the person's lifestyle based on all available information. Respond in Thai\n
    - Persona: Guess the person's persona based on all available information. Respond in Thai\n
    - Linkedin: The only one exact URL of their Linkedin profile.\n
    - Facebook: The only one exact URL of their Facebook profile.\n
    - Instagram: The only one exact URL of thier instagrame profile\n
    - Other_Social: All other social media details, including exact links. Respond in plain text\n
    - Other_information: Any additional details relevant to use for insurance marketing.\n

    Information Provided:
    {search_result_eng} \n
    {search_result_thai} \n

    Instruction: Extract these entities and provide them strictly as a valid JSON object, with proper formatting and correct field names. Keep all details in their original language (Thai) and do not translate. If no information matches an entity, return null without fabricating it. Do not include explanations, backticks, or code formatting in the output.\n
    Important: Do not generate arrays or objects in each entity in json format please use comma , instead in case that it contain many information.\n
    
    Finally, please ensure it is properly formatted with all keys enclosed in double quotes and values either in double quotes (for strings), null (if no data), or an appropriate type: 
    """
    
    ai_response = client_AI.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that extracts occupation, education, hobbies, social media profiles, and other key personal details from descriptions for marketing purposes."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    
    ai_output = ai_response.choices[0].message.content
    
    print(ai_output, "\n")
    
    try:
        json_data = json.loads(ai_output)
    except json.JSONDecodeError:
        json_data = {"error": "JSON parsing failed for additional information."}
    
    return json_data

def process_ocr_and_ai(file_path, instructions, file_name):
    with open(file_path, "rb") as f:
        image_data = f.read()

    extracted_result = client_OCR.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ]
    )

    extracted_text = ""
    if extracted_result.read is not None and extracted_result.read.blocks:
        for line in extracted_result.read.blocks[0].lines:
            extracted_text += line.text.strip()

    Extracted_Text = extracted_text.strip()

    print(Extracted_Text, '\n')

    if not Extracted_Text:
        return {"error": "No text detected in the image.", "image_path": file_name}

    response = client_AI.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": instructions.format(Extracted_Text)
            }
        ]
    )
    raw_response = response.choices[0].message.content

    try:
        json_data = json.loads(raw_response)

        # ID Card Verification
        pattern1 = r"\d{1}\s\d{4}\s\d{5}\s\d{2}\s\d{1}"
        pattern2 = r"\d{1}\s*\d{4}\s*\d{5}\s*\d{2}\s*\d{1}"
        id_card_match = re.search(pattern1, Extracted_Text)
        if not id_card_match:
            id_card_match = re.search(pattern2, Extracted_Text)

        if id_card_match:
            id_card_value = id_card_match.group().replace(" ", "")
            if thaiid.verify(id_card_value):
                json_data['ID_Card'] = id_card_value
            else:
                json_data['ID_Card'] = None
        else:
            json_data['ID_Card'] = None
        
        # Additional step: use the extracted Name and Eng_Name to get other info
        name = json_data.get('Name', None)
        eng_name = json_data.get('Eng_Name', None)

        if name:
            other_info = query_with_AI(name, eng_name if eng_name else "")
            return {"json_data": json_data, "image_path": file_name, "other_info": other_info}
        else:
            return {"json_data": json_data, "image_path": file_name, "other_info": None}

    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from OpenAI.", "image_path": file_name}

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/", methods=["GET", "POST"])
def upload_and_process():
    if request.method == "POST":
        try:
            file = request.files.get("file")
            if not file:
                return jsonify({"error": "No file uploaded."})

            document_type = request.form.get("document_type")
            if not document_type:
                return jsonify({"error": "No document type selected."})

            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            if document_type == "driving-license":
                instructions = (
                    "You are the professional in Thai driving license format for 10 years. You are here to help to processes OCR information to extract specific entities and output them as JSON. "
                    "The entities you need to extract are:\n"
                    "- License_ID: The number for Thai Driving License. It typically 8 digits coming after the words 'ฉบับที่' or 'No,', please respond without space\n"
                    "- Dob: Date of Birth in 'DD MMMM YYYY' format in Thai language. \n"
                    "- Prefix: The person's titles (e.g., นาย, น.ส., etc.). It should be exact titles from OCR text\n"
                    "- Name: The person's full name in Thai without any titles.\n"
                    "- Eng_Name: Extract only the person's full name in uppercase English letters. Do not include any titles, prefixes, or words such as 'Mr.', 'Mrs.', 'Miss', or any similar designations. The output must strictly be the person's name without these terms.\n"
                    "\nHere is the OCR text to analyze:\n"
                    "{}\n\n"
                    "Instruction: Please extract these entities and provide them strictly as a valid JSON object, "
                    "with proper formatting and correct field names. All details should remain in their original language "
                    "(Thai). Do not translate any content. Do not include any explanation, backticks, or code formatting. If you could not find the content that is match the entity return null do not mock it up. The content should come from the OCR text only.\n"
                    "Important: Extract all information exactly as presented in the OCR text. Ensure no details are missed, and preserve the full content of the input, including all numbers, names, dates, and other textual data."
                )
            elif document_type == "identification-card":
                instructions = (
                    "You are the professional in Thai identification format for 10 years. You are here to help to processes OCR information to extract specific entities and output them as JSON. "
                    "The entities you need to extract are:\n"
                    "- Dob: Date of Birth in 'DD MMMM YYYY' format in Thai language. \n"
                    "- Prefix: The person's titles (e.g., นาย, น.ส., etc.). It should be exact titles from OCR text\n"
                    "- Name: The person's full name in Thai without any titles.\n"
                    "- Eng_Name: Extract only the person's full name in uppercase English letters. Do not include any titles, prefixes, or words such as 'Mr.', 'Mrs.', 'Miss', or any similar designations. The output must strictly be the person's name without these terms.\n"
                    "- Address: The person's Thai full address including all information regarding address\n"
                    "- Province: The person province according to their address\n"
                    "- Religion: Their religion (e.g., พุทธ, คริสต์, อิสลาม)\n"
                    "\nHere is the OCR text to analyze:\n"
                    "{}\n\n"
                    "Instruction: Please extract these entities and provide them strictly as a valid JSON object, "
                    "with proper formatting and correct field names. All details should remain in their original language "
                    "(Thai). Do not translate any content. Do not include any explanation, backticks, or code formatting. If you could not find the content that is match the entity return null do not mock it up. The content should come from the OCR text only.\n"
                    "Important: Extract all information exactly as presented in the OCR text. Ensure no details are missed, and preserve the full content of the input, including all numbers, names, dates, and other textual data."
                )
            elif document_type == "car-registration":
                instructions = (
                    "You are the professional in Thai car registeration format for 10 years. You are here to help to processes OCR information to extract specific entities and output them as JSON. "
                    "The entities you need to extract are:\n"
                    "- Date_of_Registeration: The date that the car has been registed in Thai language. \n"
                    "- Car_Plate_Number: The car pleate number\n"
                    "- Car_Province: The province that car has been registed\n"
                    "- Car_Type: The car type\n"
                    "- Car_Brand: The car brand\n"
                    "- Car_Model: The car model\n"
                    "- Car_Year: The car year\n"
                    "- Car_Color: The car color\n"
                    "- Car_Chassis: The car chassis number\n"
                    "- Car_Engine_Number: The car engine number\n"
                    "- Car_CC: The car cc.\n"
                    "- Car_HP: The car hp.\n"
                    "- Car_Weight: The car weight in kilogram.\n"
                    "- Owner_Address: The owner's Thai full address including all information regarding address\n"
                    "- Owner_Province: The owner province according to their address\n"
                    "- Owner_Dob: The owner Date of Birth in 'DD MMMM YYYY' format in Thai language. \n"
                    "- Name: The owner's full name in Thai without any titles.\n"
                    "- Owner_Prefix: The owner's titles (e.g., นาย, น.ส., etc.). It should be exact titles from OCR text\n"
                    "\nHere is the OCR text to analyze:\n"
                    "{}\n\n"
                    "Instruction: Please extract these entities and provide them strictly as a valid JSON object, "
                    "with proper formatting and correct field names. All details should remain in their original language "
                    "(Thai). Do not translate any content. Do not include any explanation, backticks, or code formatting. If you could not find the content that is match the entity return null do not mock it up. The content should come from the OCR text only.\n"
                    "Important: Extract all information exactly as presented in the OCR text. Ensure no details are missed, and preserve the full content of the input, including all numbers, names, dates, and other textual data."
                )
            else:
                return jsonify({"error": "Invalid document type selected."})

            result = process_ocr_and_ai(file_path, instructions, file.filename)

            return jsonify(result)
            
        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)