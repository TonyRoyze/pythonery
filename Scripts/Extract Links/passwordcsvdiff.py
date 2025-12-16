import csv
from typing import Dict, List, Optional, Tuple

def _find_column_indices(header: List[str]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return indices for password, name, username columns if present (case-insensitive)."""
    password_col = None
    name_col = None
    username_col = None
    for idx, col in enumerate(header):
        normalized = col.strip().lower()
        if normalized == 'password':
            password_col = idx
        elif normalized in ('name', 'full_name', 'fullname', 'display_name'):
            name_col = idx
        elif normalized in ('username', 'user', 'login'):
            username_col = idx
    return password_col, name_col, username_col

def read_records_from_csv(file_path: str) -> List[Dict[str, str]]:
    """
    Read records from a CSV file.
    If a 'password' column exists, preserve optional 'name' and 'username' columns when present.
    If no header or no 'password' column, treat the first column of each row as the password.
    Returns list of dicts with keys: 'password', 'name', 'username'. Missing fields are empty strings.
    """
    records: List[Dict[str, str]] = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        if header:
            password_col, name_col, username_col = _find_column_indices(header)
            if password_col is not None:
                for row in reader:
                    if not row:
                        continue
                    password_value = row[password_col].strip() if len(row) > password_col else ''
                    name_value = row[name_col].strip() if (name_col is not None and len(row) > name_col) else ''
                    username_value = row[username_col].strip() if (username_col is not None and len(row) > username_col) else ''
                    records.append({'password': password_value, 'name': name_value, 'username': username_value})
            else:
                # No 'password' column; treat first column as password (including header as first value)
                # Include header as a row only if it exists and is not empty
                first_row = header
                if first_row and first_row[0].strip():
                    records.append({'password': first_row[0].strip(), 'name': '', 'username': ''})
                for row in reader:
                    if row and row[0].strip():
                        records.append({'password': row[0].strip(), 'name': '', 'username': ''})
        # If no header, treat as empty file
    return records

def passwords_difference(csv1_path: str, csv2_path: str, output_path: Optional[str] = None) -> None:
    """
    Print records whose password is present in csv1 but NOT in csv2.
    Optionally write results to a CSV via output_path.
    """
    records1 = read_records_from_csv(csv1_path)
    records2 = read_records_from_csv(csv2_path)

    passwords2 = {r['password'] for r in records2 if r.get('password')}

    diff_records = [r for r in records1 if r.get('password') and r['password'] not in passwords2]

    print(f"Passwords present in '{csv1_path}' but NOT in '{csv2_path}':\n")
    for r in diff_records:
        name = r.get('name', '')
        username = r.get('username', '')
        pwd = r.get('password', '')
        # Print a concise line including available fields
        if name or username:
            print(f"{pwd}\t{name}\t{username}")
        else:
            print(pwd)

    if output_path:
        # Export in the requested format: Title,URL,Username,Password,Notes,OTPAuth
        fieldnames = ['Title', 'URL', 'Username', 'Password', 'Notes', 'OTPAuth']
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in diff_records:
                title_value = r.get('name') or r.get('username') or ''
                writer.writerow({
                    'Title': title_value,
                    'URL': '',
                    'Username': r.get('username', ''),
                    'Password': r.get('password', ''),
                    'Notes': '',
                    'OTPAuth': '',
                })
        print(f"\nWrote {len(diff_records)} rows to: {output_path} (Title,URL,Username,Password,Notes,OTPAuth)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Show passwords present in CSV1 but not in CSV2, preserving optional name/username fields.")
    parser.add_argument('csv1', help='Path to first CSV (source)')
    parser.add_argument('csv2', help='Path to second CSV (exclude set)')
    parser.add_argument('-o', '--output', help='Optional path to write results as CSV')

    args = parser.parse_args()
    passwords_difference(args.csv1, args.csv2, args.output)

