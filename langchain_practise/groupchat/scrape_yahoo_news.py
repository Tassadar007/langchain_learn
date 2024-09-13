# filename: scrape_yahoo_news.py
import requests
from bs4 import BeautifulSoup
import random

# Step 1: Scrape the latest news headlines from Yahoo.com
url = "https://news.yahoo.com/"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all news articles
articles = soup.find_all('h3')

# Extract titles and links
news_list = []
for article in articles:
    link_tag = article.find('a')
    if link_tag:
        title = article.get_text()
        link = link_tag['href']
        if not link.startswith('http'):
            link = 'https://news.yahoo.com' + link
        news_list.append((title, link))

# Debugging output
print(f"Found {len(news_list)} articles.")

# Randomly select one article if the list is not empty
if news_list:
    selected_article = random.choice(news_list)
    print(selected_article)
else:
    print("No articles found.")