import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, unquote
import re 
import csv 
import argparse 

#Global Session Object 
session = requests.Session()

def login_to_lms(login_url, username, password):
    """Handles login to the LMS. This version attempts to find and use a logintoken."""
    global session
    try:
        print(f"Fetching login page: {login_url} to get logintoken...")
        login_page_resp = session.get(login_url)
        login_page_resp.raise_for_status()
        soup = BeautifulSoup(login_page_resp.text, 'html.parser')
        
        logintoken_input = soup.find('input', {'name': 'logintoken'})
        if not logintoken_input:
            print("Could not find 'logintoken' input field on the login page.")
            logintoken_value = '' 
        else:
            logintoken_value = logintoken_input['value']
            print(f"Found logintoken: {logintoken_value}")

        payload = {
            'username': username,
            'password': password,
            'logintoken': logintoken_value,
        }
        

        print(f"Attempting to POST login to: {login_url}")
        response = session.post(login_url, data=payload)
        response.raise_for_status()

        if "logout.php" in response.text or "/my/" in response.url or username in response.text:
            print("Login successful!")
            return True
        else:
            print("Login failed. Check credentials or login page structure.")
            print("Response URL after login attempt:", response.url)
            # print("Response text after login attempt (first 500 chars):", response.text[:500]) # for debugging
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error during login: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during login: {e}")
    return False

def get_resource_links_from_page(page_url, base_url):
    """Scans a given page URL for links that likely point to downloadable resources in Moodle."""
    global session
    try:
        print(f"Accessing page: {page_url} to find resource links...")
        response = session.get(page_url) 
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        resource_links = []
        for link_tag in soup.find_all('a', href=True, class_='aalink'):
            href = link_tag.get('href')
            link_text_span = link_tag.find('span', class_='instancename')
            link_text = link_text_span.text.strip() if link_text_span else ''

            if 'mod/resource/view.php' in href:
                full_url = urljoin(base_url, href)
                resource_links.append(full_url)
                print(f"Found potential resource link: {full_url} (Text: {link_text})")
        
        return list(set(resource_links))

    except requests.exceptions.RequestException as e:
        print(f"Error accessing page {page_url}: {e}")
        return []

def download_file(file_url, download_directory):
    """Downloads a file from file_url into download_directory using the session, avoiding re-downloads."""
    global session
    log_file_name = "_downloaded_log.txt"
    log_file_path = os.path.join(download_directory, log_file_name)

    try:
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r', encoding='utf-8') as lf:
                downloaded_urls = [line.strip() for line in lf.readlines()]
            if file_url in downloaded_urls:
                print(f"Skipping already downloaded file (URL found in log): {file_url}")
                return True
    except Exception as e:
        print(f"Warning: Could not read download log {log_file_path}: {e}")

    try:
        print(f"Attempting to download: {file_url}")
        r = session.get(file_url, stream=True, allow_redirects=True)
        r.raise_for_status()

        file_name = ""
        content_disposition = r.headers.get('content-disposition')
        if content_disposition:
            fname_match = re.findall(r'filename\*?=(?:UTF-8\'\')?([^;]+)', content_disposition, re.IGNORECASE)
            if fname_match:
                file_name = unquote(fname_match[-1].strip('"\\'))
        
        if not file_name:
            parsed_url_path = unquote(r.url.split('/')[-1].split('?')[0]) 
            if parsed_url_path: 
                file_name = parsed_url_path

        if not file_name:

            content_type = r.headers.get('content-type')
            extension = ''
            if content_type:

                main_type, sub_type = content_type.split('/', 1) if '/' in content_type else (content_type, '')

                if sub_type:

                    sub_type_clean = sub_type.split(';')[0].strip()
                    if sub_type_clean == 'msword': extension = '.doc'
                    elif sub_type_clean == 'vnd.openxmlformats-officedocument.wordprocessingml.document': extension = '.docx'
                    elif sub_type_clean == 'vnd.ms-powerpoint': extension = '.ppt'
                    elif sub_type_clean == 'vnd.openxmlformats-officedocument.presentationml.presentation': extension = '.pptx'
                    elif sub_type_clean == 'vnd.ms-excel': extension = '.xls'
                    elif sub_type_clean == 'vnd.openxmlformats-officedocument.spreadsheetml.sheet': extension = '.xlsx'
                    elif sub_type_clean in ['jpeg', 'jpg']: extension = '.jpg'
                    elif sub_type_clean == 'png': extension = '.png'
                    elif sub_type_clean == 'plain': extension = '.txt' 
                    elif sub_type_clean == 'zip': extension = '.zip'
                    elif sub_type_clean == 'pdf': extension = '.pdf'
                    elif sub_type_clean == 'octet-stream': extension = '.dat'
                    elif sub_type_clean == 'mpeg': extension = '.mpg'
                    elif sub_type_clean == 'mpeg2': extension = '.mp2'
                    elif sub_type_clean == 'mpeg4': extension = '.mp4'
                    else:
                        if re.match(r'^[a-zA-Z0-9]+$', sub_type_clean) and len(sub_type_clean) <= 4:
                             extension = '.' + sub_type_clean
            file_name = f"downloaded_file_{os.urandom(4).hex()}{extension if extension else '.dat'}"
            print(f"Could not determine filename reliably from headers or URL, using generated name: {file_name}")


        file_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', file_name) 
        file_name = file_name[:200]

        download_path = os.path.join(download_directory, file_name)
        

        counter = 1
        original_download_path = download_path
        while os.path.exists(download_path):
            name, ext = os.path.splitext(original_download_path)
            download_path = f"{name}_{counter}{ext}"
            counter += 1
            if counter > 100: 
                print(f"Too many existing files with similar names to {original_download_path}, skipping.")
                return False
        if counter > 1:
            print(f"File {original_download_path} already exists. Saving as {download_path}")

        print(f"Downloading as {file_name} to {download_path}...")
        with open(download_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {file_name}")


        try:
            with open(log_file_path, 'a', encoding='utf-8') as lf:
                lf.write(file_url + '\n')
        except Exception as e:
            print(f"Warning: Could not write to download log {log_file_path}: {e}")
        
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {file_url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with {file_url}: {e}")
    return False

def get_course_details(course_name_query, csv_file_path):
    """Reads the CSV file and returns the URL and code for the given course name."""
    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 3:
                    course_name_csv = row[0].strip()
                    course_url_csv = row[1].strip()
                    course_code_csv = row[2].strip()

                    if course_name_query.lower() in course_name_csv.lower():
                        print(f"Found course: {course_name_csv}, URL: {course_url_csv}, Code: {course_code_csv}")
                        return course_url_csv, course_code_csv
        print(f"Course matching '{course_name_query}' not found in {csv_file_path}")
        return None, None
    except FileNotFoundError:
        print(f"Error: Courses CSV file not found at {csv_file_path}")
        return None, None
    except Exception as e:
        print(f"Error reading CSV file {csv_file_path}: {e}")
        return None, None

def load_credentials_from_env(env_file_path):
    """Loads LMS_USERNAME and LMS_PASSWORD from a .env-like file."""
    credentials = {}
    try:
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'): # Ensure it's a key-value pair and not a comment
                    key, value = line.split('=', 1)
                    credentials[key.strip()] = value.strip().strip('"\'') # Remove surrounding quotes
        return credentials.get('LMS_USERNAME'), credentials.get('LMS_PASSWORD')
    except FileNotFoundError:
        print(f"Error: Environment file not found at {env_file_path}")
        return None, None
    except Exception as e:
        print(f"Error reading environment file {env_file_path}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Download files from an LMS course page.")
    parser.add_argument("course_name", help="The name (or partial name) of the course to download files for.")
    args = parser.parse_args()

    # --- Configuration: Base URLs and File Paths ---
    lms_base_url = 'https://sci.cmb.ac.lk/lms/' 
    login_page_url = 'https://sci.cmb.ac.lk/lms/login/index.php' 
    
    base_download_directory = 'YOUR_DOWNLOAD_DIRECTORY'
    courses_csv_path = 'YOUR_COURSES_CSV_PATH'
    env_file_path = 'YOUR_ENV_FILE_PATH'
    # --- End Configuration ---

    username, password = load_credentials_from_env(env_file_path)

    if not username or not password:
        print("Could not load credentials from env.txt. Please ensure LMS_USERNAME and LMS_PASSWORD are set in the file.")
        return

    course_page_url, course_code = get_course_details(args.course_name, courses_csv_path)

    if not course_page_url or not course_code:
        print(f"Could not find details for course '{args.course_name}'. Exiting.")
        return
    
    course_specific_download_dir = os.path.join(base_download_directory, course_code)

    if not os.path.exists(course_specific_download_dir):
        os.makedirs(course_specific_download_dir)
        print(f"Created directory: {course_specific_download_dir}")

    if not login_to_lms(login_page_url, username, password):
        print("Aborting due to login failure.")
        return
    
    print("Login seems successful, proceeding to find resource links...")

    resource_urls = get_resource_links_from_page(course_page_url, lms_base_url)

    if not resource_urls:
        print(f"No resource links found on {course_page_url} for course {args.course_name} ({course_code}).")
        return

    print(f"Found {len(resource_urls)} potential resource(s) for {args.course_name} ({course_code}). Starting download...")


    download_count = 0
    for file_url_to_download in resource_urls:
        if download_file(file_url_to_download, course_specific_download_dir):
            download_count += 1
    
    print(f"Finished. Downloaded {download_count} out of {len(resource_urls)} files for {args.course_name} ({course_code}).")

if __name__ == '__main__':
    main()