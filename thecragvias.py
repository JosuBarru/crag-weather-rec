from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import pandas as pd


vias = list()
graduaciones = list()

def scrap(url):
    page = requests.get(url)
    html = BeautifulSoup(page.content, 'html.parser')

    entrada = html.find('div', {'class': 'node-listview hide-archived'})
    if entrada is  None:
        return

    info_via = entrada.find_all('div', {'class': 'route'})
    if info_via is None:
        return

    for i in info_via:
        nombre_via = i.find('span', {'class': 'primary-node-name'})
        graduacion = i.find('span', {'class': 'gb3'})
        if(nombre_via is None):
            continue
        nombre_via=nombre_via.text
        if(graduacion is None):
            continue
        graduacion=graduacion.text
        print(nombre_via,' : ' , graduacion)
        vias.append(nombre_via)
        graduaciones.append(graduacion)
        

class Crawler:

    def __init__(self, urls=[]):
        self.visited_urls = []
        self.urls_to_visit = urls

    def download_url(self, url):
        return requests.get(url).text

    def get_linked_urls(self, url, html):
        soup = BeautifulSoup(html, 'html.parser')
        cont = soup.find('div', {'class': 'node-listview hide-archived'})
        if cont is not None:
            for i in cont.find_all('div', {'class': 'name'}):
                link = i.find('a')
                if link is not None:
                    path = link.get('href')
                    if path is None:
                        continue
                    elif path.startswith('/'):
                        path = urljoin(url, path)
                    yield path

    def add_url_to_visit(self, url):
        if url not in self.visited_urls and url not in self.urls_to_visit:
            self.urls_to_visit.append(url)

    def crawl(self, url):
        html = self.download_url(url)
        for url in self.get_linked_urls(url, html):
            self.add_url_to_visit(url)

    def run(self):
        cont=0
        while self.urls_to_visit and cont<=1000:
             url = self.urls_to_visit.pop(0)
             print(url) 
             scrap(url)
             self.crawl(url)
             self.visited_urls.append(url)
             cont=cont+1
             print(cont)

if __name__ == '__main__':
    Crawler(urls=['https://www.thecrag.com/es/escalar/spain/pamplona-vitoria-gasteiz-area']).run()
    df = pd.DataFrame({'vias': vias, 'graduaciones': graduaciones})
    df.to_csv('vias.csv', index=False)