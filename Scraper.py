from bs4 import BeautifulSoup
import requests
import re

class Scraper:
    def __init__(self) -> None:
        self.trainingdata = {}
        self.testingdata = {}
        self.urls = {}
        self.empires = []

    def scrape_training(self):
        for empire in self.urls["Training"]:
            urls = self.urls["Training"][empire]
            for url in urls:
                request = requests.get(url)
                soup = BeautifulSoup(request.content, "html.parser")
                content = ""
                for tag in soup.find_all('p'):
                    text = tag.get_text()
                    if empire.lower() in text.lower():
                        content += text

                content = self.format_data(content)


                return content


    def scrape_testing(self):
        pass

    def remove_phrases(self, content):
        content = content.lower()
        with open("Resources/Data/Removalphrases.txt", "r") as file:
            file = [line.strip().replace("\n", "") for line in file.readlines()]
            for line in file:
                print(line, line.lower() in content)
                content = re.sub(line.lower(), " ", content)
        return content
        
    def format_data(self, content):
        content = re.sub("  +", " ", content)
        content = re.sub("\n", " ", content)
        content = self.remove_phrases(content)
        return content

    def to_csv(self, filename):
        pass

    def set_empires(self):
        with open("Resources/Data/URLs.txt", "r") as file:
            empires = file.readline().split(" ")
            training_urls = {}
            testing_urls = {}
            for empire in empires:
                self.trainingdata[empire] = []
                self.testingdata[empire] = []
                training_urls[empire] = []
                testing_urls[empire] = []
                self.empires.append(empire)
            self.urls["Training"] = training_urls
            self.urls["Testing"] = testing_urls

    def format_urls(self):
        with open("Resources/Data/URLs.txt", "r") as file:
            file = file.readlines()
            current_empire = None
            current_dataset = None
            for line in file:
                line = line.strip().replace('\n', '')
                if line in ("Training", "Testing"):
                    current_dataset = line
                elif line in self.empires:
                    current_empire = line
                elif self.is_link(line):
                    self.urls[current_dataset][current_empire].append(line)                       
            
    def is_link(self, link):
        if link[:4] == 'http':
            if link[4] == ':' or link[5] == ':':
                return True
        return False


