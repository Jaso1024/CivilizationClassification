import unittest
from Scraper import Scraper

class tester(unittest.TestCase):
    def setUp(self) -> None:
        self.scraper = Scraper()
        self.scraper.set_empires()
    
    def test_is_link(self):
        self.assertTrue(self.scraper.is_link("https://www.google.com"))
        self.assertTrue(self.scraper.is_link("http://www/google.com"))
        self.assertFalse(self.scraper.is_link("www/google.com"))

    def test_format_urls(self):

        self.scraper.format_urls()

unittest.main()