# 使用selenuim模拟浏览器爬取数据
from selenium import webdriver

chrome_options = webdriver.ChromeOptions()
chrome_options.headless = True
chrome = webdriver.Chrome(options=chrome_options)
chrome.get('https://www.baidu.com')
