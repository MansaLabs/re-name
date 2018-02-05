"""
Scripts to scrap information from websites
"""

# # %% using scrapy
# import scrapy

# class BlogSpider(scrapy.Spider):
#     name = 'blogspider'
#     start_urls = ['https://blog.scrapinghub.com']

#     def parse(self, response):
#         for title in response.css('h2.entry-title'):
#             yield {'title': title.css('a ::text').extract_first()}

#         for next_page in response.css('div.prev-post > a'):
#             yield response.follow(next_page, self.parse)

# %% using beautiful soup
# import urllib2
from bs4 import BeautifulSoup
import requests

url = input("Enter a website to extract the URL's from: ")

r  = requests.get("http://" +url)

data = r.text

soup = BeautifulSoup(data, "lxml")

href_all = []
for link in soup.find_all('a'):
    href = link.get('href')
    if href[:4] == '/url':
        # print(href.split('//'))
        href_all += href.split('//')

href_trim =[]
for i, items in enumerate(href_all):
    # print (items)
    if items[:4] != '/url' and items[:4] != 'webc':
        link_ = items.split('&sa')[0]
        if  link_[-3:] != 'lnk':
            href_trim.append(items.split('&sa')[0])

# now we have urls from a search result
for items in href_trim:
    print(items)
print('lenght of soup:',len(href_trim))