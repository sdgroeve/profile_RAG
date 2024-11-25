import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin

def main():
    base_url = 'https://ai.ugent.be'
    url = 'https://ai.ugent.be/people/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    researchers = []

    # Find the container with class 'people-grid'
    people_grid = soup.find('div', class_='people-grid')

    # Find all 'a' tags with class 'grid-card' within the people grid
    grid_cards = people_grid.find_all('a', class_='grid-card')

    for card in grid_cards:
        # Get the profile link and make it absolute
        profile_link = card.get('href')
        profile_link = urljoin(base_url, profile_link)

        # Get the image tag
        img_tag = card.find('img')

        # Get image URL from 'data-echo' attribute
        image_url = img_tag.get('data-echo') if img_tag else None
        image_url = urljoin(base_url, image_url)

        # Get name from 'alt' attribute of the image tag
        name = img_tag.get('alt') if img_tag else None

        # If name is not found in 'alt', try to get it from figcaption
        if not name:
            figcaption = card.find('figcaption')
            name = figcaption.get_text(strip=True) if figcaption else None

        researcher = {
            'name': name,
            'profile_link': profile_link,
            'image_url': image_url
        }

        researchers.append(researcher)

    # Save the data to a JSON file
    with open('researchers.json', 'w', encoding='utf-8') as f:
        json.dump(researchers, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
