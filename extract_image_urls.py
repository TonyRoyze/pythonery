import re

def extract_image_urls(input_file, output_file):
    try:
        with open(output_file, 'w') as out_file:
            pass
        
        chunk_size = 1024 * 1024  
        pattern = re.compile(r'src="(.*?)">')
        
        with open(input_file, 'r', encoding='utf-8') as file:
            chunk = file.read(chunk_size)
            
            while chunk:
                urls = pattern.findall(chunk)
                
                if urls:
                    with open(output_file, 'a', encoding='utf-8') as out_file:
                        for url in urls:
                            out_file.write(url + '\n')
                
                chunk = file.read(chunk_size)
                
        print(f"URLs have been extracted and saved to {output_file}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = input("Site.txt")
    output_file = input("links.txt")
    
    extract_image_urls(input_file, output_file)