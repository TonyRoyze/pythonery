# DFL.py - Download From LMS

`DFL.py` is a Python script designed to automate the download of course materials from a Moodle-based Learning Management System (LMS).

## Features

*   Logs into the LMS.
*   Fetches resource links from a specified course page.
*   Downloads files to a structured directory based on the course code.
*   Avoids re-downloading already fetched files by maintaining a log.
*   Handles various file types and attempts to determine correct file extensions.

## Prerequisites

*   Python 3
*   The following Python libraries (install via pip):
    *   `requests`
    *   `beautifulsoup4`

```bash
pip install requests beautifulsoup4
```

## Configuration

Before running the script, you need to configure a few things:

1.  **LMS Credentials (`env.txt`)**: Create a new text file named `env.txt` in the location you specified for `env_file_path` within `DFL.py`. Add your LMS username and password to this file in the following format, replacing the placeholder values with your actual credentials:

    ```plaintext:%2Fpath%2Fto%2Fyour%2Fenv.txt
    LMS_USERNAME='YOUR_USERNAME'
    LMS_PASSWORD='YOUR_PASSWORD'
    ```
    **Important:** Ensure there are no extra spaces around the `=` sign and that your username and password are enclosed in single or double quotes if they contain special characters (though the script attempts to strip them).

2.  **Course Information (`courses.csv`)**: Create a new CSV file named `courses.csv` in the location you specified for `courses_csv_path` within `DFL.py`. This file should contain the course codes and names for which you want to download materials. The format of the CSV file should be as follows:
    ```csv
    DataStructuresAndAlgorithms, https://sci.cmb.ac.lk/lms/course/view.php?id=176, CS3008
    DataVisualization, https://sci.cmb.ac.lk/lms/course/view.php?id=667, DS3001
    DataEthics, https://sci.cmb.ac.lk/lms/course/view.php?id=668, DS3002
    ```
    Each line should contain the course name, the URL of the course page, and the course code. IMPORTANT: Ensure there are no extra spaces in the course name and course code.

3.  **Script Configuration (Inside `DFL.py`)**: You **must** adjust the following placeholder variables directly within the `DFL.py` script's `main` function. These tell the script where to find your configuration files and where to save downloaded materials.
    *   `lms_base_url`: The base URL of your LMS (e.g., `https://sci.cmb.ac.lk/lms/`). This is the main address of the Moodle site. You might need to change this if you are targeting a different Moodle instance.
    *   `login_page_url`: The direct URL to the LMS login page (e.g., `https://sci.cmb.ac.lk/lms/login/index.php`). This is the page where you enter your username and password. This might also need to be changed for different Moodle instances.
    *   `base_download_directory`: **Replace `'YOUR_DOWNLOAD_DIRECTORY'`** with the absolute path to the root directory on your computer where course-specific folders will be created.
    *   `courses_csv_path`: **Replace `'YOUR_COURSES_CSV_PATH'`** with the absolute path to your `courses.csv` file.
    *   `env_file_path`: **Replace `'YOUR_ENV_FILE_PATH'`** with the absolute path to your `env.txt` file.

    Specifically, these lines in `DFL.py` (around lines 234-238) define these paths and **require your input**:

    ```python
    # --- Configuration: Base URLs and File Paths ---
    lms_base_url = 'https://sci.cmb.ac.lk/lms/' 
    login_page_url = 'https://sci.cmb.ac.lk/lms/login/index.php' 
    
    base_download_directory = 'YOUR_DOWNLOAD_DIRECTORY' # <-- IMPORTANT: SET THIS PATH
    courses_csv_path = 'YOUR_COURSES_CSV_PATH'       # <-- IMPORTANT: SET THIS PATH
    env_file_path = 'YOUR_ENV_FILE_PATH'             # <-- IMPORTANT: SET THIS PATH
    # --- End Configuration ---
    ```


