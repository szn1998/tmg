import scrapy
from scrapy.http import Request, HtmlResponse
from urllib.parse import urljoin
from music_crawl.items import MusicCrawlItem

from selenium import webdriver


def get_html(url):
    """
    return html content via headless browser
    """
    browser = get_browser(url)
    page_source = browser.page_source
    browser.quit()

    return page_source


def get_browser(url):
    """
    Headless browser object
    """
    options = webdriver.FirefoxOptions()
    options.headless = True
    browser = webdriver.Firefox(options=options)
    browser.get(url)

    return browser


class Tibet3Spider(scrapy.Spider):
    name = "tibet3"
    allowed_domains = ["music.tibet3.com"]
    start_urls = ["http://music.tibet3.com/music/playlist"]

    @staticmethod
    def full_url(base_url="http://music.tibet3.com", relative_url=None):
        return urljoin(base_url, relative_url)

    def parse(self, response):
        genres = response.xpath('//h2[@class="lm"]/span/a')
        for genre in genres:
            genre_link = genre.xpath("./@href").get()
            yield response.follow(genre_link, callback=self.parse_genre)

    def parse_genre(self, response):
        yield response.follow(response.url, callback=self.parse_album_page)

        next_page = response.xpath("//a[contains(text(), 'འོག་ངོས')]/@href")

        if next_page:
            yield response.follow(
                next_page.get(), callback=self.parse_album_page
            )

    def parse_album_page(self, response):
        albums = response.xpath(
            '//ul[@class="m_grup_list"]/li/p[@class="dec"]/a'
        )
        for album in albums:
            album_link = album.xpath("./@href").get()
            album_link = self.full_url(relative_url=album_link)
            yield response.follow(
                album.xpath("./@href").get(), callback=self.parse_album_record
            )

    def parse_album_record(self, response):
        url = response.url
        response = get_html(response.url)
        response = HtmlResponse(url=url, body=response, encoding="utf-8")
        mp3 = response.xpath('//a[@class="jp-playlist-item-free"]')
        # mp3 = response.selector.xpath("//script")
        for music in mp3:
            item = MusicCrawlItem()
            mp3_link = music.xpath("./@href").get()
            mp3_link = self.full_url(relative_url=mp3_link)
            item["link"] = mp3_link
            yield item
