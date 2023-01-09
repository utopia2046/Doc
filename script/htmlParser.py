# https://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/

from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen

#zpid = "18429834"
#url = "http://www.zillow.com/homes/" + zpid + "_zpid/"
url = "https://en.wikipedia.org/wiki/English_units"

#html = urlopen("https://www.bing.com")

response = requests.get(url)
html = response.content

print(html)

soup = BeautifulSoup(html, "html.parser")
print(soup.title.string)
print(soup.prettify())

# %conda install html5lib
html5libResults = BeautifulSoup(html, "html5lib").find_all('div', attrs={"id":"toc"})
print(len(html5libResults))
htmlParserResults = BeautifulSoup(html, "html.parser").find_all('div', attrs={"class":"thumb"})
print(len(htmlParserResults))
lxmlResults = BeautifulSoup(html, "lxml").find_all('div', attrs={"id":"toc"})
print(len(lxmlResults))
