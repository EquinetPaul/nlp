from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
tqdm.pandas()
import time, random

filename_save = "discours_macron.csv"
nb_pages = 446

# Config WebDriver

## Créer un DataFrame pandas avec deux colonnes: url, title
discours_df = pd.DataFrame(columns=["url", "title"])

base_url = "https://www.vie-publique.fr"
search_url = base_url + "/discours/recherche?search_api_fulltext_discours=&sort_by=field_date_prononciation_discour&field_intervenant_title=Jacques+Chirac&field_intervenant_qualite=&field_date_prononciation_discour_interval%5Bmin%5D=&field_date_prononciation_discour_interval%5Bmax%5D=&form_build_id=form-Ot7Ms03LottLQo-JNZWj59AIFl1eaYWNKZky8suBIIo&form_id=views_exposed_form&page="

chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(executable_path="C:/Users/equinetpa/Desktop/PFE/scrap/chromedriver.exe", options=chrome_options)

# Step 1: Get Speeches URL

for page_num in tqdm(range(0,nb_pages)):  # Itérer sur toutes les pages
    full_url = search_url + str(page_num)
    driver.get(full_url)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    elements = soup.find_all('h3', class_='fr-card__title')

    # Itération sur les éléments trouvés
    for el in elements:
        link = el.find('a')
        title = link.text.strip()
        url = "https://www.vie-publique.fr" + link['href']

        new_row = pd.DataFrame({'url': [url], 'title': [title]})
        discours_df = pd.concat([discours_df, new_row], ignore_index=True)
    # time.sleep(random.randint(1,2)

##  Sauvegarder le DataFrame dans un fichier CSV
discours_df.to_csv(filename_save, sep=";", index=False, encoding="utf-8")

# Step 2: Get Speeches Content
def get_content(url, driver):
    driver.get(url)

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    try:
        # Récupération des éléments de la classe 'fr-tag fr-tag--green-emeraude'
        tags = soup.find_all(class_='fr-tag fr-tag--green-emeraude')
        tag = tags.pop(0).text.strip() if tags else None
        keywords = ', '.join([t.text.strip() for t in tags])
    except:
        tags = None
        keywords = None

    # Récupération du premier élément de la classe 'datetime'
    datetime_element = soup.find(class_='datetime')
    datetime = datetime_element.text.strip() if datetime_element else None

    # Récupération du premier élément de la classe spécifiée
    context_element = soup.find(class_='field field--name-field-circonstance field--type-string field--label-hidden field__item')
    context = context_element.text.strip() if context_element else None

    try:
        # Récupération du contenu de la classe 'vp-discours-content'
        content = soup.find(class_='vp-discours-content')
        if content:
            content.h2.decompose()  # Suppression du titre h2
            speech = content.get_text(separator='\n').strip()
        else:
            speech = None
    except:
        speech = None

    # time.sleep(random.randint(1,2))

    return tag, keywords, datetime, context, speech

discours_df[['tag', 'keywords', 'datetime', 'context', 'speech']] = discours_df['url'].progress_apply(lambda url: pd.Series(get_content(url, driver)))

# Leave and Save
## Fermez le navigateur après avoir terminé
driver.quit()

## Sauvegarder le DataFrame dans un fichier CSV
discours_df.to_csv(filename_save, sep=";", index=False, encoding="utf-8")
