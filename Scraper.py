from bs4 import BeautifulSoup
import requests

class Scraper:
    def __init__(self) -> None:
        self.trainingdata = {}
        self.testingdata = {}

    def scrape_training(self):
        pass
    
    def get_empires(self):
        with open("Resources/Data/URLs.txt", "r") as file:
            empires = file.readline().split(" ")
            for empire in empires:
                self.trainingdata[empire] = []
                self.testingdata[empire] = []

    def format_urls(self):
        pass
            
    def scrape_testing(self):
        pass

    def format_data(self):
        pass

    def to_csv(self, filename):
        pass

if __name__ == '__main__':
    scraper = Scraper()
    scraper.get_empires()