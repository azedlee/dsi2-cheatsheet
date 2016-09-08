### Web Scraping
#=====================================================================================================
from scrapy.selector import Selector
from scrapy.http import HtmlResponse
import requests

# Example selector with commonly used tags
Selector(text=HTML).xpath('/html/body/ul/li[@class = "alpha"][@id = "id"]/text()').extract()

# Scrap all the links on a website, ex. for www.datatau.com
//td[@class='title'][2]/a/@href

# Get Request
response = requests.get("http://www.datatau.com")
HTML = response.text
# view the first 500 characters of the HTML index document for DataTau
HTML[0:500]

# Contains is a (if 'string' in x) statement
//td[@class='subtext']/span[contains(@id,'score')]/text()

# Looking for the more link
//a[text()="More"]/@href

# Other ways
best1        = Selector(text=HTML).xpath('/html/body/div/p/a[@class="bestof-link"]')
nested_best1 = best1.xpath('./span[@class="bestof-text"]/text()').extract()
print nested_best1

# Through command line
mkdir scrapy_projects # Creates a new directory
scrapy startproject craigslist # Within the directory, pre-generates all the necessary files
"""
Creates pre-generated files from scrapy

craigslist/
    scrapy.cfg
    craigslist/
        __init__.py
        items.py
        pipelines.py
        settings.py
        spiders/
            __init__.py
            ...

scrapy.cfg: the project configuration file
craigslist/: the project’s python module, you’ll later import your code from here.
craigslist/items.py: the project’s items file.
craigslist/pipelines.py: the project’s pipelines file.
craigslist/settings.py: the project’s settings file.
craigslist/spiders/: a directory where you’ll later put your spiders.
"""
scrapy shell http://sfbay.craigslist.org/search/sfc/apa # Any webpage works for the shell
scrapy crawl craigslist -o apts.csv # -o saves file to apts.csv
#=====================================================================================================