import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin
import time

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

        # Request the profile page
        profile_response = requests.get(profile_link)
        profile_soup = BeautifulSoup(profile_response.content, 'html.parser')

        # Extract email
        email_tag = profile_soup.find('a', href=lambda href: href and 'mailto:' in href)
        email = email_tag.get_text(strip=True) if email_tag else None
        researcher['email'] = email

        # Extract phone number
        phone_tag = profile_soup.find('a', href=lambda href: href and 'tel:' in href)
        phone = phone_tag.get_text(strip=True) if phone_tag else None
        researcher['phone'] = phone

        # Extract research unit
        unit = None
        # Find all spans and look for one containing 'research unit'
        for span in profile_soup.find_all('span'):
            if 'research unit' in span.get_text(strip=True).lower():
                # Get the next sibling 'a' tag
                next_sibling = span.find_next_sibling('a')
                if next_sibling:
                    unit = next_sibling.get_text(strip=True)
                break
        researcher['research_unit'] = unit

        # Extract personal website
        website = None
        contact_div = profile_soup.find('div', class_='person-contact')
        if contact_div:
            # Exclude 'mailto:' and 'tel:' links
            website_links = contact_div.find_all('a', href=lambda href: href and not href.startswith('mailto:') and not href.startswith('tel:') and 'ugent.be' not in href)
            if website_links:
                website = website_links[0].get('href')
        researcher['website'] = website

        # Extract biography
        bio_tag = profile_soup.find('div', class_='person-bio')
        bio = bio_tag.get_text(strip=True) if bio_tag else None
        researcher['bio'] = bio

        # Extract keywords
        keywords = None
        keywords_tag = profile_soup.find('div', class_='person-keywords')
        if keywords_tag:
            strong_tag = keywords_tag.find('strong')
            if strong_tag:
                strong_tag.decompose()  # Remove 'Keywords:' label
            keywords = keywords_tag.get_text(strip=True)
        researcher['keywords'] = keywords

        # Extract key publications
        publications = []
        publications_section = profile_soup.find('div', class_='person-publications')
        if publications_section:
            key_pubs_header = publications_section.find('strong', string=lambda x: x and 'Key publications' in x)
            if key_pubs_header:
                ul_tag = key_pubs_header.find_next_sibling('ul')
                if ul_tag:
                    pubs = ul_tag.find_all('li')
                    for pub in pubs:
                        publications.append(pub.get_text(strip=True))
        researcher['publications'] = publications

        researchers.append(researcher)

        # Sleep to be polite to the server
        time.sleep(1)

    # Save the data to a JSON file
    with open('researchers.json', 'w', encoding='utf-8') as f:
        json.dump(researchers, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
