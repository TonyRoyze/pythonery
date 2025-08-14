import re

with open('links.html', 'r', encoding='utf-8') as f:
    html = f.read()

magnet_links = re.findall(r'href=["\'](magnet:\?.*?)["\']', html)

with open('magnet_links.txt', 'w', encoding='utf-8') as out:
    for link in magnet_links:
        out.write(link + '\n')