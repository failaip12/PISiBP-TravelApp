import datetime
import os
import random
import string
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import functools
from pathlib import Path
import pickle
from tqdm import tqdm

import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from faker import Faker
from mysql.connector import Error  # OBRISI PRAZNA POLJA
from playwright.sync_api import sync_playwright
from utils import (BED_NUM, TYPE, companyGPT, dodajRandomAktivnost,
                   dodajRandomGrad, hotelGPT, plotTools, roundStars,
                   translateElement)
from config import PATH_DATA
from tenacity import retry, stop_after_attempt, wait_exponential

"""
    
NOTES TO SELF:
TESTIRATI GENERATORE

    """
global link
link = 'https://www.worldometers.info/geography/7-continents/'          

def getDriver():#na kraju prevesti celokupni df
    
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch()
    page = browser.new_page()
    return page, browser, playwright

def getContinents():
    
    kontinenti = pd.DataFrame(columns = ['naziv'])
    
    
    page, browser, playwright = getDriver()
    global link
    page.goto(link)
    page.wait_for_timeout(2000)
    content = page.content()
    soup = BeautifulSoup(content,features="lxml")
    names = []
    for a in soup.find_all('table', attrs={'class':'table'}):
        for name in a.find_all('a'):
            names.append(name.text) #names sadrzi engl kontinente
        for name in names:
        #continents.append(cyrillic_to_latin(tss.google(name, fr, to)))
            kontinenti.loc[len(kontinenti.index)] = name

    browser.close()
    playwright.stop()
    return kontinenti
    
def getContinentsAndCountries():    
    global link
    kontinenti = getContinents()
    page, browser, playwright = getDriver()
    
    drzave = pd.DataFrame(columns = ['naziv','kontinent'])

    for name in kontinenti['naziv'].values:
        temp = name.replace(" ","-")
        newlink = link + '/'+name.lower().replace(' ','-')+'/'

        page.goto(newlink)
        page.wait_for_timeout(2000)
        content = page.content()
        soup = BeautifulSoup(content,features="lxml")
        for a in soup.find_all('table', attrs={'class':'table'}):
            howmuch = 15 if name == 'Europe' else 3                                               #koliko drzava po kontinentu?
            
            for country in a.find_all('td'):
                if not howmuch:
                    break
                
                if all(x.isalpha() or x.isspace() for x in country.text):
                    
                    drzave.loc[len(drzave.index)] = [country.text,kontinenti.loc[kontinenti['naziv']==name,'naziv'].values[0]]
                    howmuch-=1
    browser.close()
    playwright.stop()
    return kontinenti,drzave
    
    
def get_city_data(args):
    name, drzave, newlink = args
    page, browser, playwright = getDriver()
    city_data = []
    
    try:
        clink = newlink + name.lower().replace(' ','-')
        page.goto(clink)
        page.wait_for_timeout(1000)  # Reduced wait time
        content = page.content()
        
        soup = BeautifulSoup(content, features="lxml")
        for a in soup.find_all('table', attrs={'class':'wpr-table'}):
            howmuch = 5 if drzave.loc[drzave["naziv"]==name,"kontinent"].values[0] == 'Europe' else 5              
            for city in a.find_all('th')[2:]:
                if not howmuch:
                    break
                if all(x.isalpha() or x.isspace() for x in city.text):
                    city_data.append([
                        city.text,
                        drzave.loc[drzave['naziv']==name,'naziv'].values[0],
                        drzave.loc[drzave['naziv']==name,'kontinent'].values[0]
                    ])
                    howmuch-=1
    finally:
        browser.close()
        playwright.stop()
    
    return city_data

def cache_to_disk(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_dir = Path(PATH_DATA) / ".cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{func.__name__}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        result = func(*args, **kwargs)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
    return wrapper

@cache_to_disk
def getAllGeography():
    kontinenti, drzave = getContinentsAndCountries()
    newlink = 'https://worldpopulationreview.com/countries/cities/'
    
    # Prepare arguments for parallel processing
    args_list = [(name, drzave, newlink) for name in drzave['naziv'].values]
    
    gradovi = pd.DataFrame(columns=['naziv','drzava','kontinent'])
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(get_city_data, args_list))
    
    # Flatten results and add to DataFrame
    for city_data in results:
        for city_info in city_data:
            gradovi.loc[len(gradovi.index)] = city_info
    
    gradovi.to_csv(os.path.join(PATH_DATA, 'cities.csv'),index=None)
    
    
    
    
    
    gradovi['naziv']=gradovi['naziv'].apply(translateElement)
    gradovi['drzava']=gradovi['drzava'].apply(translateElement)
    gradovi['kontinent']=gradovi['kontinent'].apply(translateElement)
    gradovi.to_csv(os.path.join(PATH_DATA, 'gradovi.csv'),index=None)
    return gradovi,drzave,kontinenti
    
    
    
    
def getAllDFs():
    gradovi,drzave,kontinenti = getAllGeography()
    #gradovi = pd.read_csv(os.path.join(PATH_DATA, 'gradovi.csv'))
    hoteli = pd.DataFrame(columns=["naziv","grad"])
    for index, val in gradovi['naziv'].items():

        imena = hotelGPT(val)
        for ime in imena:
            hoteli.loc[len(hoteli.index)] = [ime,val]
            print(ime)
            
    
    hoteli.to_csv(os.path.join(PATH_DATA, 'hoteli.csv'),index=None)

    return hoteli,gradovi,drzave,kontinenti



def generateRooms():   #testirati sumu broja soba
    hoteli = pd.read_csv(os.path.join(PATH_DATA, 'hoteli.csv'))
    np.random.seed(5)
    mu = 75
    sigma = 25
    hoteli['br_soba'] = np.random.normal(mu, sigma, hoteli.shape[0])*hoteli['zvezdice']
    hoteli['br_soba']=hoteli['br_soba'].apply(round)
    hoteli.to_csv(os.path.join(PATH_DATA, 'hoteli.csv'),index = None)
    

def hotelStarsDistribution():   #koristiti kasnije openAI da generise opis hotela
    
    np.random.seed(5)
    mu = 3.5
    sigma = 0.5
    
    try:    
        hoteli = pd.read_csv(os.path.join(PATH_DATA, 'hoteli.csv'))
        
    except:
        hoteli,_,_,_ = getAllDFs()
        
    hoteli['zvezdice'] = np.random.normal(mu, sigma, hoteli.shape[0])
    hoteli['zvezdice'] = hoteli['zvezdice'].apply(roundStars)
    hoteli.to_csv(os.path.join(PATH_DATA, 'hoteli.csv'),index = None)
    
def hotelGenerateAddress():
    try:
        hoteli = pd.read_csv(os.path.join(PATH_DATA, 'hoteli.csv'))
    except:
        hoteli,_,_,_ = getAllDFs()
    fake = Faker()
    cutSpaces = np.vectorize(lambda x: x.replace('  ',' ')) #[''.join(y) for y in x[0:len(x.split(' '))-1]]
    cutAptNums = np.vectorize(lambda x: (' '.join([''.join(y) for y in x.split(' ')[0:len(x.split(' '))-1]]) if x.split(' ')[len(x.split(' '))-1].isnumeric() else x))
    arr = np.array([' '.join([y if not ('Suite' in y or 'Apt.' in y) else '' for y in fake.address().split('\n')[0].split(' ') ]) for i in range(len(hoteli.index))],dtype=str)
    addresses = cutAptNums(cutSpaces(arr)) 
        #this is why we love python
    hoteli['adresa'] = addresses
    hoteli.to_csv(os.path.join(PATH_DATA, 'hoteli.csv'),index=None)

def PlotGeneratedInfos():
    hoteli = pd.read_csv(os.path.join(PATH_DATA, 'hoteli.csv'))
    gradovi = pd.read_csv(os.path.join(PATH_DATA, 'gradovi.csv'))
    plotTools(what=hoteli,case=1,title="Raspodela zvezdica hotela",x="Broj zvezdica",y="Kolicina",path = "hotel_stars_normal.png")
    plotTools(what=gradovi,case=2,title="Raspodela kolicine hotela po zemljama",x="Drzava",y="Broj hotela",path = "country_to_no_hotels.png")
    plotTools(what=gradovi,case=3,title="Raspodela kolicine hotela po kontinentima",x="Kontinent",y="Broj hotela",path = "continent_to_no_hotels.png")
    #plotTools(what=hoteli['zvezdice'],case=1,title="Raspodela zvezdica hotela",x="Broj zvezdica",y="Kolicina",path = "hotel_stars_normal.png")
    plotTools(what=hoteli,case=4,title="Raspodela soba po hotelima",x="Hoteli",y="Broj soba",path = "rooms_to_hotels.png")

def sobaParamGen(base=10):
    sobe = pd.DataFrame(columns=['tip','br_kreveta','opis','gen_cena'])


    for i in list(TYPE.keys()): #da vrati samo tip,broj kreveta, pocetnu cenu
        for j in BED_NUM:
            kreveti_multiplier = 1 if j<2 else 0.67        

            sobe.loc[len(sobe.index)] = [i,j,TYPE[i][0],round(TYPE[i][1]*j*kreveti_multiplier * base,2)]       
            
    sobe.to_csv(os.path.join(PATH_DATA, 'sobe.csv'),index = None)

def generatePrevoznik():
    prevoz = pd.DataFrame(columns=['tip','tip_komp'])
    prevoz['tip'] = ['airplane', 'bus', 'cruise', 'train']
    prevoz['tip_komp'] = companyGPT()
    prevoz['cena'] = [random.choice(range(1000,3000)),random.choice(range(100,250)),random.choice(range(250,950)),random.choice(range(100,150))]
            
    prevoz.loc[len(prevoz.index)] = ['lično vozilo', 'samostalni prevoz',0]
    prevoz.to_csv(os.path.join(PATH_DATA, 'prevoz.csv'),index=None)

    
    
    
    
def batch_process(df, batch_size=1000):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size]

def generatePonude():
    """
    aranzman(aran_id,naziv,krece,vraca,nap,smestaj_id,p_id)
    
    naziv->
    
    """
    gradovi = pd.read_csv(os.path.join(PATH_DATA, 'gradovi.csv'))
    hoteli = pd.read_csv(os.path.join(PATH_DATA, 'hoteli.csv'))
    prevoz = pd.read_csv(os.path.join(PATH_DATA, 'prevoz.csv'))
    aranzman = pd.DataFrame(columns = ["naziv","krece","vraca","smestaj","p_id"])
    hoteli['tmp'] = 1
    prevoz['tmp'] = 1
    prevoz['p_id'] = prevoz.index+1
    prevoz['prevod'] = ["avionom","autobusom","krstarenje/brodom","vozom","samostalni prevoz"]
    prevoz['tmp']=1

    temp=pd.DataFrame()
    temp['naziv'] = hoteli['naziv']
    temp['smestaj_id'] = temp.index+1
    aranzman = pd.merge(hoteli, prevoz, on=['tmp'])
    
    def startDateGenerator(margin,min_date,max_date): #uzmi random dan za start iz dana iz dates array gde vazi da zadnji dan meseca-start nije vece od margin(trajanje putovanja)
        maxed_date   = datetime.datetime.strptime(max_date, '%Y-%m-%d') - datetime.timedelta(days=int(margin))
        #minimized_date = datetime.strptime(min_date, '%Y-%m-%d') + pd.DateOffset(days=margin)
        
        dates = pd.date_range(min_date,maxed_date,freq='d').to_list()
        
        start_date = random.choice(dates)
        #end_date = datetime.strptime(start_date, '%Y-%m-%d') + pd.DateOffset(days=margin)
        
        return start_date + datetime.timedelta(hours=random.choice(range(1,20)))
    """
    def endDateGenerator(margin,min_date,max_date):
        
        minimized_date = datetime.strptime(min_date, '%Y-%m-%d') + pd.DateOffset(days=margin)
        
        dates = pd.date_range(minimized_date,max_date,freq='d').to_list()
        
        end_date = random.choice(dates)
        #end_date = datetime.strptime(start_date, '%Y-%m-%d') + pd.DateOffset(days=margin)
        
        return end_date
    """
    
    def endDateGenerator(margin,start_date):
        return start_date + datetime.timedelta(days=int(margin)) + datetime.timedelta(hours=random.choice(range(0,3)))
    #x = datetime.datetime(2018, 6, 1)

    #print(x.strftime("%B")
    #df['NewCol'] = df.apply(lambda x: segmentMatch(x['TimeCol'], x['ResponseCol']), axis=1)
    timeline = pd.DataFrame()
    timeline['broj_dana'],timeline['tmp'] = ['3','5','7','10','14'],1
    meseci = pd.DataFrame()
    meseci['mesecMin'],meseci['mesecMax'],meseci['tmp'] = ['2025-1-1','2025-6-1','2025-7-1','2025-8-1','2025-9-1','2025-10-1'],['2025-1-31','2025-6-30','2025-7-31','2025-8-31','2025-9-30','2025-10-31'],1
    meseci['mesec'],meseci['mpr'] = ['1','6','7','8','9','10'],["Januar","Jun","Jul","Avgust","Septembar","Oktobar"]
    aranzman = pd.merge(aranzman, timeline, on=['tmp'])
    aranzman = pd.merge(aranzman, meseci, on=['tmp'])
    aranzman=aranzman.merge(temp,on=['naziv'],how='left')
    aranzman = aranzman.drop('tmp', axis=1)
    aranzman['datum_pocetka'] = aranzman.apply(lambda x: startDateGenerator(x['broj_dana'],x['mesecMin'],x['mesecMax']),axis=1)
    aranzman['datum_zavrsetka'] = aranzman.apply(lambda x: endDateGenerator(x['broj_dana'],x['datum_pocetka']),axis=1)
    aranzman = aranzman.drop(columns=['mesecMax','mesecMin'], axis=1)

    aranzman['godina'] = aranzman['datum_pocetka'].dt.strftime('%Y')
    aranzman['m_str'] = aranzman['datum_pocetka'].dt.strftime('%')
    #hoteli.loc[hoteli['naziv']==aranzman['naziv'],"grad"].values() + 
    aranzman['ime'] = aranzman['grad'] + " " + aranzman['mpr'] + " "+ aranzman['godina'] + " " + aranzman['naziv'] + " " + aranzman['prevod']+ " " + aranzman['broj_dana'] + " dana"
    
    # Process in batches
    for batch in batch_process(aranzman):
        batch.to_csv(os.path.join(PATH_DATA, "aranzmani.csv"), 
                    mode='a', 
                    header=(not os.path.exists(os.path.join(PATH_DATA, "aranzmani.csv"))),
                    index=None)

def generateAktivnosti():
    df = pd.DataFrame(columns = ["naziv"])
    
    for x in ["Setnja po gradu","Obilazak nacionalnog parka","Poseta muzeju","Nocenje","Fakultativni izleti","Slobodno vreme- obilazak lokalnog soping centra",
                "Obilazak obliznjih lokaliteta","Organizovani nocni provod"]:
        df.loc[len(df.index)]=x;
        
    df.to_csv(os.path.join(PATH_DATA, "aktivnosti.csv"),index=None);
    
    
def smestajImaAktivnost():#g_id	akt_id	smestaj_id	

        
    gradovi = pd.read_csv(os.path.join(PATH_DATA, "gradovi.csv")).index+1
    akt = pd.DataFrame(columns = ["g_id","akt_id","smestaj_id"])
    akt['smestaj_id'] = pd.read_csv(os.path.join(PATH_DATA, "hoteli.csv")).index+1
    akt=akt.fillna('0')
    akt['g_id'] = akt['g_id'].apply(dodajRandomGrad)
    akt['akt_id'] = akt['akt_id'].apply(dodajRandomAktivnost)
    akt.to_csv(os.path.join(PATH_DATA, "aktivnosti_u_gradu.csv"),index=None)
    #akt['g_id'] = 
    
def generateImaAktivnost():
    decompose = lambda x: [1 for i in range(int(x))]
    
    aran = pd.read_csv(os.path.join(PATH_DATA, "aranzmani.csv"))
    aran['broj_dana'] = aran['broj_dana'].apply(lambda x: x-1)

    akt = pd.read_csv(os.path.join(PATH_DATA, "aktivnosti.csv"))
    ids = akt.index+1
    aran=aran.drop(columns=['cena','naziv','grad','br_soba','adresa','tip','tip_komp','p_id','prevod','mesec','mpr','datum_pocetka','datum_zavrsetka','godina','m_str','ime','zvezdice'])
    aran['aran_id'] = aran.index+1
    aran['akt_id']=aran['broj_dana'].apply(decompose)
    ima = aran.explode('akt_id')
    ima = ima.drop(columns=['broj_dana','smestaj_id'])
    ima['akt_id'] = np.random.choice(ids, ima.shape[0])
    ima.to_csv(os.path.join(PATH_DATA, "ima_aktivnost.csv"),index=None)
    

def generateRandomRezervacije(n):
    faker = Faker()
    
    # Create data in bulk instead of list comprehension
    rez = pd.DataFrame({
        'ime': pd.Series([faker.first_name() for _ in range(n)]),
        'prezime': pd.Series([faker.last_name() for _ in range(n)]),
        'br_kartice': pd.Series([''.join(random.choices(string.ascii_lowercase, k=8)) for _ in range(n)]),
        'email': pd.Series([f"{faker.last_name()}@gmail.com" for _ in range(n)]),
        'broj_odr': pd.Series(np.random.randint(0, 4, n)),
        'broj_dece': pd.Series(np.random.randint(0, 4, n)),
        'cena': pd.Series(np.random.randint(100, 501, n)),
        'kom': pd.Series(np.random.randint(1, 6, n)),
        'kontakt': pd.Series([''.join(random.choices(string.ascii_lowercase, k=10)) for _ in range(n)]),
        'aran_id': pd.Series(np.random.randint(1, 50001, n)),
        'broj_soba': pd.Series(np.random.randint(1, 6, n))
    })
    
    rez.to_csv(os.path.join(PATH_DATA, "rand_rez.csv"), index=None)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_page_content(page, url):
    page.goto(url)
    page.wait_for_timeout(1000)
    return page.content()

def generator():
    print("Starting data generation...")
    with tqdm(total=9, desc="Generating data") as pbar:
        getAllDFs()
        pbar.update(1)
        
        hotelStarsDistribution()
        pbar.update(1)
        
        generateRooms()
        pbar.update(1)
        
        hotelGenerateAddress()
        pbar.update(1)
        
        sobaParamGen()
        PlotGeneratedInfos()
        pbar.update(1)
        
        generatePrevoznik()
        generatePonude()
        pbar.update(1)
        
        generateAktivnosti()        
        pbar.update(1)
        
        smestajImaAktivnost()
        pbar.update(1)
        
        generateImaAktivnost()
        generateRandomRezervacije(150)
        pbar.update(1)


start = time.time()
generator()
end = time.time()
print(end - start)