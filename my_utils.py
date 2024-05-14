import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.robotparser import RobotFileParser
from difflib import SequenceMatcher
import time
from sklearn.linear_model import LinearRegression
import requests
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif 
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, r2_score
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler, RobustScaler, Binarizer
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
import numpy as np
import requests
import re
import json
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from scipy import stats
from sklearn.metrics import confusion_matrix


## Chat GPT Begin
def read_sos_csv(year, folder = 'data'):
    file_path = f'../{folder}/SOS_{year}.csv'
    
    if year in range(2011, 2015):
        df = pd.read_csv(file_path)
    elif year in range(2015, 2019):
        df = pd.read_csv(file_path, usecols=['Team', 'SOS'])
    elif year == 2019:
        df = pd.read_csv(file_path, delimiter='\t')
    else:
        raise ValueError("Year must be between 2011 and 2019.")
    
    df = df[['Team', 'SOS']]
    df = df.dropna()
    
    return df

def create_college_alternatives_dict(college_data):
    unique_college_list = college_data['college'].unique()
    special_college_mappings = {
    'florida international': ['fiu'],
    "st mary's ca": ["saint mary's college"],
    'seattle': ['seattle university'],
    'maryland-baltimore county': ['umbc'],
    'nevada-las vegas': ['unlv'],
    'kennesaw': ['kennesaw state'],
    'college of charleston': ['charleston'],
    'citadel': ['the citadel'],
    'virginia military institu': ['vmi'],
    'virginia commonwealth': ['vcu'],
    'alabama-birmingham': ['uab'],
    'nebraska-omaha': ['omaha'],
    'texas-rio grande valley': ['utrgv'],
    'texas-arlington': ['uta'],
    'florida atlantic': ['fau'],
    'unc wilmington': ['uncw'],
    'florida gulf coast': ['fgcu'],
    'unc greensboro': ['uncg'],
    'pennsylvania': ['penn'],
    'massachusetts': ['umass'],
    'texas christian': ['tcu'],
    'cs fullerton': ['cal state fullerton'],
    'texas-san antonio': ['utsa'],
    'southern mississippi': ['southern miss'],
    'wisconsin-milwaukee': ['milwaukee'],
    'cs bakersfield': ['cal state bakersfield'],
    'southern california': ['usc'],
    'nc state': ['north carolina state'],
    'massachusetts-lowell': ['umass lowell', 'umass-lowell'],
    'central florida': ['ucf'],
    'brigham young': ['byu'],
    'presbyterian': ['presbyterian college'],
    'mississippi': ['ole miss'],
    'cal poly san luis obispo': ['cal poly'],
    'cs northridge': ['cal state northridge'],
    'miami fl': ['miami (fl)', 'miami'],
    'louisiana-lafayette': ['louisiana'],
    'se louisiana': ['southeastern louisiana'],
    'texas pan american': ['utpa'],
    'illinois-chicago': ['uic'],
    'arkansas-little rock': ['little rock'],
    'texas-pan american': ['utrgv'],
    'nevada-las vegas': ['unlv'],
    'college of charleston': ['charleston'],
    'louisiana-monroe': ['ulm'],
    'citadel': ['the citadel'],
    'alabama-birmingham': ['uab'],
    'texas-arlington': ['ut arlington', 'uta'],
    'florida atlantic': ['fau'],
    'texas-rio grande valley': ['utrgv'],
    'nebraska-omaha': ['omaha'],
    'louisiana-lafayette': ['louisiana'],
    'illinois-chicago': ['uic'],
    'utah valley state': ['utah valley']
    }
    
    college_alternatives_dict = {}
    for college in unique_college_list:
        cleaned_college = clean_text(college)
        alternatives = [college]
        if cleaned_college in special_college_mappings or college in special_college_mappings:
            alternatives.extend([name for name in special_college_mappings[cleaned_college] if name not in alternatives])
        alternatives = [clean_text(alternative) for alternative in alternatives] 
        college_alternatives_dict[college] = alternatives
    return college_alternatives_dict

def clean_text(text):
    # The regular expression [^\w\s'-] will keep word characters (\w), 
    # whitespace characters (\s), dashes (-), and apostrophes (').
    cleaned_text = re.sub(r"[^\w\s'-]", '', text.replace('\xa0', ' '))
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text.lower()

# Function to match college names
def match_college_name(df_college_name, sos_colleges):
    df_college_name_cleaned = clean_text(df_college_name)
    best_match = None
    highest_ratio = 0.8
    for sos_college in sos_colleges:
        sos_college_cleaned = clean_text(sos_college)
        match_ratio = SequenceMatcher(None, df_college_name_cleaned, sos_college_cleaned).ratio()
        if match_ratio > highest_ratio:
            highest_ratio = match_ratio
            best_match = sos_college
    return best_match

# Function to map special college names

def match_and_add_sos_info(college_data, sos_years=range(2011, 2020)):
    # Load SOS data for all years
    is_allowed, delay = check_scraping_and_crawl_delay('http://warrennolan.com/')
    if is_allowed == True:
        print('Scraping allowed for SOS Website')
    else:
        print('Scraping not allowed for SOS Website')
    sos_data = {year: read_sos_csv(year) for year in sos_years}
    
    # Create the college alternatives dictionary
    college_alternatives_dict = create_college_alternatives_dict(college_data)

    # Iterate through each row in college_data
    for index, row in college_data.iterrows():
        year = row['Year']
        college_name = row['college']
        
        # Check if the year is in the sos_data
        if year in sos_data:
            # Get the list of alternative names for the college
            alternative_names = college_alternatives_dict.get(college_name, [college_name])

            best_match = None
            best_match_ratio = 0  # Renamed for consistency

            # Iterate through each alternative name
            for alt_name in alternative_names:
                # Use match_college_name function to find the best match
                current_match = match_college_name(alt_name, sos_data[year]['Team'].unique())
                if current_match:
                    current_ratio = SequenceMatcher(None, clean_text(alt_name), clean_text(current_match)).ratio()
                    if current_ratio > best_match_ratio:
                        best_match_ratio = current_ratio
                        best_match = current_match

            # Update the college_data DataFrame with the best match and corresponding SOS
            if best_match:
                college_data.at[index, 'matched_college_name'] = best_match
                college_data.at[index, 'SOS'] = sos_data[year].loc[sos_data[year]['Team'] == best_match, 'SOS'].iloc[0]

    return college_data



def make_request(url, max_retries=5):
    for _ in range(max_retries):
        response = requests.get(url)
        if response.status_code == 429:
            print("Received 429 Too Many Requests. Retrying after a delay...")
            time.sleep(10)  # Adjust the delay time as needed
        elif response.status_code == 200:
            return response
        else:
            print(f"Error: {response.status_code} for {url}. Retrying...")
            time.sleep(5)  # Adjust the delay time as needed
    return None

def check_scraping_and_crawl_delay(url, user_agent='*'):
    """
    Check if the robots.txt file of a website allows scraping for a specific user agent and returns the crawl delay.

    Parameters:
    url (str): The URL of the website.
    user_agent (str): The user agent to check for. Defaults to '*', applicable to all user agents.

    Returns:
    tuple: (bool, int or None) - True if scraping is allowed, False otherwise, and the crawl delay if specified.
    """
    rp = RobotFileParser()
    rp.set_url(f"{url.rstrip('/')}/robots.txt")

    try:
        rp.read()
        # Check if scraping is allowed for the user agent and return crawl delay
        is_allowed = rp.can_fetch(user_agent, url)
        crawl_delay = rp.crawl_delay(user_agent)
        return is_allowed, crawl_delay
    except Exception as e:
        print(f"Error reading robots.txt: {e}")
        return False, None
    

# Function to match college names and add the associated name and SOS from the SOS dataset







def get_college_stats(data_for_cleaning):
    base_url = "https://www.baseball-reference.com"
    is_allowed, crawl_delay = check_scraping_and_crawl_delay(base_url)
    n = len(data_for_cleaning)
    data_for_cleaning['college_WOBA'] = pd.Series([0.0] * n, dtype=float)
    data_for_cleaning['walk_to_strikeout'] = pd.Series([0.0] * n, dtype=float)
    data_for_cleaning['age'] = pd.Series([0] * n, dtype=int)
    data_for_cleaning['college'] = pd.Series([''] * n, dtype=str)
    data_for_cleaning['Year'] = 0
    data_for_cleaning['SB'] = 0
    data_for_cleaning['Position'] = 0
    if is_allowed is True:
        print(f"Crawl delay: {crawl_delay} seconds")
        
        rows_to_drop = []

        for i in range(len(data_for_cleaning)):
            id = data_for_cleaning.loc[i, 'key_bbref_minors']
            url = f"{base_url}/register/player.fcgi?id={id}"

            if crawl_delay is not None:
                print(f"Waiting {crawl_delay} seconds based on robots.txt rules...")
                time.sleep(crawl_delay)

            response = make_request(url)

            while response is None:
                print("Waiting for a successful response...")
                time.sleep(10)
                response = make_request(url)

            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table')

            if not tables:
                print(f"No tables found for {id}. Possibly a pitcher.")
                rows_to_drop.append(i)
                continue

            table_data = pd.read_html(str(tables[0]))[0]
            subset_data = table_data[table_data['Lev'] == 'NCAA'].reset_index(drop=True)
            total_rows = subset_data.shape[0]

            if total_rows > 0:
                final_row = subset_data.iloc[total_rows - 1, :]
                required_columns = ["BB", "IBB", "HBP", "H", "2B", "3B", "HR", "AB", "SF", "HBP", "SO"]
                if not set(required_columns).issubset(final_row.index):
                    print(f"{id} is a pitcher. Removing from data_for_cleaning.")
                    rows_to_drop.append(i)
                    continue

                final_row[required_columns] = pd.to_numeric(final_row[required_columns], errors='coerce').fillna(0)

                try:
                    college_WOBA = (0.690 * (final_row['BB'] - final_row['IBB']) +
                                    0.722 * final_row['HBP'] +
                                    0.89 * final_row['H'] +
                                    1.27 * final_row['2B'] +
                                    1.62 * final_row['3B'] +
                                    2.10 * final_row['HR']) / \
                                   (final_row['AB'] + final_row['BB'] - final_row['IBB'] +
                                    final_row['SF'] + final_row['HBP'])

                    walk_to_strikeout = final_row['BB'] / final_row['SO']
                    age = final_row['Age']
                    college = final_row['Tm']

                    data_for_cleaning.at[i, 'college_WOBA'] = college_WOBA
                    data_for_cleaning.at[i, 'walk_to_strikeout'] = walk_to_strikeout
                    data_for_cleaning.at[i, 'age'] = age
                    data_for_cleaning.at[i, 'college'] = college
                    data_for_cleaning.at[i, 'Year'] = final_row['Year']
                    data_for_cleaning.at[i, 'SB'] = final_row['SB']

                except ZeroDivisionError:
                    print(f"ZeroDivisionError for {id}. Adding index to the list of rows to drop.")
                    rows_to_drop.append(i)

            else:
                print(f"No NCAA data found for {id}. Possibly a pitcher.")
                rows_to_drop.append(i)
            print(f'Player number {i}')
            data_for_cleaning.to_csv('updated.csv')

        data_for_cleaning.drop(index=rows_to_drop, inplace=True)
        print(data_for_cleaning)
    else:
        print("Unable to determine crawl delay from robots.txt. Aborting.")

    return data_for_cleaning

def get_player_positions(data_for_cleaning):
    base_url = "https://www.baseball-reference.com/register/player.fcgi?id="
    robots_url = "https://www.baseball-reference.com"

    is_allowed, crawl_delay = check_scraping_and_crawl_delay(robots_url)
    if not is_allowed:
        print("Scraping not allowed as per robots.txt")
        return data_for_cleaning

    data_for_cleaning['Position'] = pd.Series([''] * len(data_for_cleaning), dtype=str)

    for i, row in data_for_cleaning.iterrows():
        player_id = row['key_bbref_minors']
        url = f"{base_url}{player_id}"

        if crawl_delay:
            time.sleep(crawl_delay)

        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            position_tag = soup.find('strong', text=lambda x: x and 'Position' in x)
            if position_tag and position_tag.next_sibling:
                positions = position_tag.next_sibling.strip().split(',')
                # Take only the first position listed
                first_position = positions[0].strip().split(' ')[0] if positions else ''
                data_for_cleaning.at[i, 'Position'] = first_position
            else:
                print(f"No position data found for player with id {player_id}.")
        else:
            print(f"Request for player with id {player_id} failed with status code {response.status_code}.")
        data_for_cleaning.to_csv('updated.csv')

    return data_for_cleaning





class Phase2(BaseEstimator, TransformerMixin):
    """
    A transformer class for preprocessing data, with capabilities for handling missing data, 
    downsampling, one-hot encoding, and dropping specific columns and rows based on values.
    """
    def __init__(self, na_method='drop', balance_method='random_sampling', random_state = 42, response_column=None, 
                 binary=False, categorical_columns=False, drop_columns=None, index_column=None, 
                 k_means_to_consider=[5,10,20,25,30], columns_to_use_in_balancing=None, 
                 columns_to_consider_for_dropping={}, downsample=True, file_output= None, dict_output= None):
        """
        Initialize the Phase2 transformer.
        Parameters:
            na_method (str): Method for handling missing data.
            balance_method (str): Method for balancing the dataset.
            response_column (str): Name of the response column for binary classification.
            binary (bool): Flag to indicate if the response variable is binary.
            categorical_columns (list): List of categorical columns for one-hot encoding.
            drop_columns (list): List of columns to drop from the dataset.
            index_column (str): Column to set as the index.
            k_means_to_consider (list): List of k values to consider for KMeans clustering.
            columns_to_use_in_balancing (list): Columns to consider for balancing the dataset.
            columns_to_consider_for_dropping (dict): Columns and their respective values for dropping rows.
            downsample (bool): Flag to enable or disable downsampling.
            random_state (int): Random state for reproducibility.
        """
        self.na_method = na_method
        self.balance_method = balance_method
        self.response_column = response_column
        self.binary = binary
        self.categorical_columns = categorical_columns
        self.drop_columns = drop_columns
        self.index_column = index_column
        self.k_means_to_consider = k_means_to_consider
        self.columns_to_use_in_balancing = columns_to_use_in_balancing
        self.columns_to_consider_for_dropping = columns_to_consider_for_dropping
        self.downsample = downsample
        self.random_state = random_state
        self.dropped_indices = []
        self.file_output = file_output
        self.dict_output = dict_output
        self.random_state = random_state
    def fit(self, X, y=None):
        self.dropped_indices = []
        return self

    def transform(self, X):
        """
    Transforms the data by applying the preprocessing steps defined in the class.
    This includes reindexing, handling missing data, dropping rows based on specific values,
    one-hot encoding, and optionally balancing the dataset through downsampling.

    Parameters:
        X (DataFrame): The DataFrame to be transformed.

    Returns:
        DataFrame: Transformed DataFrame.
        dict: Dictionary containing the balancing scores, if downsampling is performed.
        list: List of indices of rows that were dropped during the transformation process.
        """
        
        data = X.copy()
        data = self.reindex_and_drop(data)

        # Track dropped indices in each step
        data = self.handle_missing_data(data)
        data = self.drop_rows_based_on_values(data)
        
        data = self.one_hot_encode(data)
        if  self.categorical_columns:
            print(f'Used One Hot Encoder with { self.categorical_columns}')
        if self.downsample and self.binary and self.response_column:
            data, score_dict = self.handle_binary_response(data)
            if self.dict_output is not None:
                print(f'Sending score_dict to {self.dict_output}')
                with open(self.dict_output, 'w') as file:
                    json.dump(score_dict, file, indent=4)
            data = self.drop_unnamed_columns(data)
            if self.file_output is not None:
                print(f'Sending data to {self.file_output}')
                data.to_csv(self.file_output, index = True)
            if self.dropped_indices:
                return data, score_dict, self.dropped_indices
            else:
                return data, score_dict
        else:
            data = self.drop_unnamed_columns(data)
            if self.file_output is not None:
                print(f'Sending data to {self.file_output}')
                data.to_csv(self.file_output, index = True)
            return data, self.dropped_indices

    def handle_missing_data(self, data):
        """
        Handles missing data in the DataFrame according to the specified method.
        Options include dropping rows with missing values, filling missing values with the mean,
        or using regression to impute missing values.
        Parameters:
        data (DataFrame): The DataFrame with potential missing values.
        Returns:
        DataFrame: The DataFrame after handling missing values.
        """
        missing_values_count = data.isna().sum().sum()
        if self.na_method == 'drop':
            before_drop = data.index.tolist()
            data = data[data.notna().all(axis=1)]
            print(f'Dropped {missing_values_count} missing values')
            after_drop = data.index.tolist()
            self.dropped_indices.extend([idx for idx in before_drop if idx not in after_drop])
        elif self.na_method == 'avg':
            imputer = SimpleImputer(strategy='mean')
            original_index = data.index  # Store the original index
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=original_index)
            print(f'Used Mean Imputer for {missing_values_count} NA values')
        elif self.na_method == 'regression' and self.response_column:
            data = self.regression_imputation(data)
            print(f'Used Regression for {missing_values_count} NA Values')
        else:
            print("Invalid method or missing response column for regression imputation.")
            return data

        return data

    def regression_imputation(self, data):
        """
        Performs regression imputation on the DataFrame. Uses linear regression to predict
        and fill missing values in the response column based on other features.
        Parameters:
        data (DataFrame): The DataFrame with missing values in the response column.
        Returns:
        DataFrame: The DataFrame after performing regression imputation.
        """
        non_na_data = data.dropna(subset=[self.response_column])
        na_data = data[data.isna().any(axis=1)]
        reg = LinearRegression().fit(non_na_data.drop(columns=[self.response_column]), non_na_data[self.response_column])
        predicted_values = reg.predict(na_data.drop(columns=[self.response_column]))
        na_data[self.response_column].fillna(pd.Series(predicted_values, index=na_data.index), inplace=True)
        return pd.concat([non_na_data, na_data])
    def handle_binary_response(self, data):
        """
        Handles binary response data by balancing the dataset. This includes evaluating and
        applying random sampling or KMeans sampling for downsampling. It also calculates 
        and returns the performance scores of the applied methods to help in selecting 
        the best downsampling approach.

        Parameters:
            data (DataFrame): The DataFrame containing the binary response data.

        Returns:
            DataFrame: The balanced DataFrame after applying the selected downsampling method.
            dict: A dictionary containing performance scores of the evaluated downsampling methods.
        """
        score_dict = {}
        data_for_cross_score = data.copy()
        if self.columns_to_use_in_balancing is not None:
            data_for_cross_score = data_for_cross_score[self.columns_to_use_in_balancing + [self.response_column]]
        if self.binary and self.response_column:
            # Evaluate random sampling
            balanced_data_random = self.random_sampling(data_for_cross_score)
            scores_random = self.evaluate_model(balanced_data_random)
            score_dict['random_sampling'] = scores_random

            # Evaluate KMeans sampling
            _, scores_kmeans = self.kmeans_sampling(data_for_cross_score)
            for k, score in scores_kmeans.items():
                score_dict[f'kmeans_{k}_equal'] = score

        # Choose the method with the best combination of high mean and low std
        def score_method(key):
            mean_score = score_dict[key]['mean']
            std_score = score_dict[key]['std']
            # Adjust these weights as needed
            weight_for_mean = 1.0
            weight_for_std = -0.5  # Negative weight since lower std is better
            return mean_score * weight_for_mean + std_score * weight_for_std

        best_method = max(score_dict, key=score_method)
        if best_method == 'random_sampling':
            balanced_data = self.random_sampling(data)
            print('Using random_sampling to downsample for balanced data as it has the best combination of high mean score and low std')
        else:
            k_best = int(best_method.split('_')[1])
            balanced_data = self.create_balanced_data_using_kmeans(data, k_best, strategy='equal')
            print(f'Using kmeans with {k_best} clusters (equal method) to downsample for balanced data as it has the best combination of high mean score and low std')

        return balanced_data, score_dict

    def random_sampling(self, data):
        class_counts = data[self.response_column].value_counts()
        min_class = class_counts.idxmin()
        n_min_class = class_counts.min()

        data_min = data[data[self.response_column] == min_class]
        data_max = data[data[self.response_column] != min_class].sample(n=n_min_class, random_state=self.random_state)

        return pd.concat([data_min, data_max])
    def kmeans_sampling(self, data, n_clusters=None):
        class_counts = data[self.response_column].value_counts()
        min_class = class_counts.idxmin()
        n_min_class = class_counts.min()

        if n_clusters is None:
            n_clusters = [n_min_class] + self.k_means_to_consider

        scores = {}
        for k in n_clusters:
            print(f'Doing kmeans clustering with {k} clusters')
            data_max = data[data[self.response_column] != min_class]

            # Ensure consistent data ordering
            data_max_sorted = data_max.sort_index()

            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10, max_iter=300).fit(data_max_sorted.drop(self.response_column, axis=1))

            # Strategy: Equal Points from Each Cluster
            data_max_sample_equal = self.sample_equal_points_from_each_cluster(data_max_sorted, kmeans, n_min_class // k, class_counts, random_state=self.random_state)
            balanced_data_equal = pd.concat([data[data[self.response_column] == min_class], data_max_sample_equal])
            score_equal = self.evaluate_model(balanced_data_equal)
            scores[k] = score_equal

        best_k = max(scores, key=lambda k: scores[k]['mean'])
        best_balanced_data = self.create_balanced_data_using_kmeans(data, best_k, strategy='equal')

        return best_balanced_data, scores

    def sample_equal_points_from_each_cluster(self, data, kmeans, n_points_per_cluster, class_counts, random_state=42):
        labels = kmeans.labels_
        sampled_indices = []

        # Initialize random state
        rng = np.random.default_rng(random_state)

        # First pass: sample n_points_per_cluster from each cluster if possible
        for i in range(kmeans.n_clusters):
            indices = np.where(labels == i)[0]
            if len(indices) >= n_points_per_cluster:
                sampled_indices.extend(rng.choice(indices, n_points_per_cluster, replace=False))

        # Additional sampling if needed to balance classes
        while True:
            class_counts_sampled = data.iloc[sampled_indices][self.response_column].value_counts()
            if class_counts_sampled.min() == class_counts.min():
                break  # Classes are balanced

            for i in range(kmeans.n_clusters):
                # Check class balance after each addition
                class_counts_sampled = data.iloc[sampled_indices][self.response_column].value_counts()
                if class_counts_sampled.min() == class_counts.min():
                    break  # Classes are balanced

                indices = np.where(labels == i)[0]
                # Filter out already sampled indices
                new_indices = list(set(indices) - set(sampled_indices))

                if new_indices:
                    sampled_indices.append(rng.choice(new_indices, 1)[0])

        return data.iloc[sampled_indices]

    def create_balanced_data_using_kmeans(self, data, k, strategy='equal', random_state=42):
        class_counts = data[self.response_column].value_counts()
        min_class = class_counts.idxmin()

        # Ensure consistent data ordering
        data_max = data[data[self.response_column] != min_class].sort_index()

        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300).fit(data_max.drop(self.response_column, axis=1))

        if strategy == 'closest':
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data_max.drop(self.response_column, axis=1))
            data_max_sample = data_max.iloc[closest]
        else:  # strategy == 'equal'
            data_max_sample = self.sample_equal_points_from_each_cluster(data_max, kmeans, class_counts[min_class] // k, class_counts, random_state=random_state)

        return pd.concat([data[data[self.response_column] == min_class], data_max_sample])

    def reindex_and_drop(self, data):
        if self.index_column and self.index_column in data.columns:
            print(f'Reindexing by {self.index_column}')
            data = data.set_index(self.index_column)
        if self.drop_columns:
            print(f'Dropping {self.drop_columns} as they are not useful for learning')
            data = data.drop(columns=self.drop_columns, errors='ignore')
        return data

    def one_hot_encode(self, data):
        """
        Applies one-hot encoding to the categorical columns of the data.
        Parameters:
            data (DataFrame): Input DataFrame to encode.
        Returns:
        DataFrame: Transformed DataFrame with one-hot encoded columns.
        """
        if self.categorical_columns:
            encoder = OneHotEncoder(sparse=False, drop='first')
            categorical_data = data[self.categorical_columns]
            encoded_data = encoder.fit_transform(categorical_data)
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(self.categorical_columns))

            # Set the index of encoded_df to match the original DataFrame's index
            encoded_df.index = data.index

            # Drop original categorical columns and concatenate with encoded DataFrame
            data = data.drop(columns=self.categorical_columns, errors='ignore')
            return pd.concat([data, encoded_df], axis=1)
        else:
            # If no categorical columns, return the original data
            return data
    def evaluate_model(self, data):
        # Split the data into features and target
        X = data.drop(self.response_column, axis=1)
        y = data[self.response_column]

        # Define the number of folds and the random state
        num_folds = 5
        random_state = self.random_state

        # Initialize lists to store the cross-validation scores
        scores = []

        # Create StratifiedKFold object for cross-validation
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)

        # Perform manual cross-validation
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Define the model to be used for each fold
            model = LogisticRegression(random_state=random_state)

            # Fit the model on the training data
            model.fit(X_train, y_train)

            # Evaluate the model on the test data and store the score
            score = model.score(X_test, y_test)
            scores.append(score)

        # Calculate the mean and standard deviation of the scores
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        return {'mean': mean_score, 'std': std_score}
    def drop_rows_based_on_values(self, data):
        rows_dropped_count = 0

        if self.columns_to_consider_for_dropping:
            for column, values_to_drop in self.columns_to_consider_for_dropping.items():
                original_row_count = len(data)
                before_drop = data.index.tolist()
                data = data[~data[column].isin(values_to_drop)]
                after_drop = data.index.tolist()
                rows_dropped_in_iteration = original_row_count - len(data)
                rows_dropped_count += rows_dropped_in_iteration

                # Store the indices of dropped rows
                dropped_indices_in_iteration = [idx for idx in before_drop if idx not in after_drop]
                self.dropped_indices.extend(dropped_indices_in_iteration)
                print(f'Dropped {rows_dropped_in_iteration} rows from column {column} based on values {values_to_drop}')

        print(f'Total dropped rows: {rows_dropped_count} that had values in {self.columns_to_consider_for_dropping}')
        return data
    def drop_unnamed_columns(self, data):
        """
        Drops columns from a DataFrame that have names starting with 'Unnamed'.

        Parameters:
        data (DataFrame): The DataFrame from which to drop columns.

        Returns:
        DataFrame: A new DataFrame with the specified columns dropped.
        """
        # Identify columns that start with 'Unnamed'
        unnamed_columns = [col for col in data.columns if col.startswith('Unnamed')]

        # Drop these columns
        return data.drop(columns=unnamed_columns, errors='ignore')
import warnings

# Add this line at the beginning of your script to filter out the specified warnings
warnings.filterwarnings("ignore", message="Features .* are constant.", category=UserWarning)

class Phase3(BaseEstimator, TransformerMixin):
    def __init__(self, k_to_consider = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100],random_state = 44, pca_params=None, scale_normalize=False, num_features_to_consider=None, response_variable=None, classification=True, regression=False, try_pca=True, feature_reduction=True, numeric_columns=[], file_output = None, dict_output= None):
        self.pca_params = pca_params
        self.scale_normalize = scale_normalize
        self.num_features_to_consider = num_features_to_consider
        self.response_variable = response_variable
        self.classification = classification
        self.regression = regression
        self.numeric_columns = numeric_columns
        self.k_to_consider = k_to_consider
        self.best_method = None
        self.best_pca_model = None
        self.best_pca_scaler = None
        self.best_pca_normalizer = None
        self.cat_columns_to_keep = None
        self.total_columns_to_keep = None
        self.k_scaler = None
        self.k_normalizer = None
        self.score_dict = {}
        self.best_pca_scoring = (0,np.inf)
        self.best_k_scoring =  (0,np.inf)
        self.numeric_features = None
        self.categorical_features = None
        self.file_output = file_output
        self.dict_output = dict_output
        self.random_state = random_state

    def fit(self, X, y=None):
        data = X.copy()
        if self.response_variable:
            self.response_data = data.pop(self.response_variable)
        else:
            raise ValueError("Response variable not provided.")

        self.numeric_features = data[self.numeric_columns]
        self.categorical_features = data.drop(columns=self.numeric_columns)
        if self.pca_params is not None:
            self.choose_best_pca(numeric_features, categorical_features)
            self.determine_best_k_for_feature_selection(self.numeric_features, self.categorical_features, k_params =  self.k_to_consider, response_var =  self.response_data)
            self.best_method = 'PCA' if self.compare_scores(self.best_pca_model, self.best_k_scoring) else 'KBest'
        else:
            self.determine_best_k_for_feature_selection(self.numeric_features, self.categorical_features, k_params =  self.k_to_consider, response_var =  self.response_data)
            self.best_method = 'KBest'
        return self
    def transform(self, X):
        np.random.seed(42)
        data = X.copy()
        if self.best_method == 'PCA':
            final_data = self.prepare_data_for_phase_4(data, self.response_variable, self.numeric_columns, self.best_pca_scaler, self.best_pca_normalizer, self.best_pca_model, self.cat_columns_to_keep)
            print('Returning data ready for modeling, pca object, dictionary of scores, scaler, normalizer, and categorical columns to keep')
            if self.dict_output is not None:
                print(f'Sending score_dict to {self.dict_output}')
                with open(self.dict_output, 'w') as file:
                    json.dump(self.score_dict, file, indent=4)
            if self.file_output is not None:
                print(f'Sending data to {self.file_output}')
                final_data.to_csv(self.file_output, index = True)
            return final_data, self.best_pca_model, self.score_dict, self.best_pca_scaler, self.best_pca_normalizer, self.cat_columns_to_keep
        elif self.best_method == 'KBest':
            final_data = self.prepare_data_for_phase_4(data, self.response_variable, self.numeric_columns, self.k_scaler, self.k_normalizer, pca_model = None, selected_features = self.total_columns_to_keep)
            print('Returning final data ready for modeling, columns to keep, scoring dictionary, scaler, and normalizer')
            if self.dict_output is not None:
                print(f'Sending score_dict to {self.dict_output}')
                with open(self.dict_output, 'w') as file:
                    json.dump(self.score_dict, file, indent=4)
            if self.file_output is not None:
                print(f'Sending data to {self.file_output}')
                final_data.to_csv(self.file_output, index = True)
            return final_data, self.total_columns_to_keep, self.score_dict, self.k_scaler, self.k_normalizer
            


    def compare_scores(self, pca_score, k_score):
        pca_mean, pca_std = pca_score
        k_mean, k_std = k_score
        # Adjust these weights as needed
        weight_for_mean = 1.0
        weight_for_std = -0.5  # Negative since lower std is better
        pca_final_score = pca_mean * weight_for_mean + pca_std * weight_for_std
        k_final_score = k_mean * weight_for_mean + k_std * weight_for_std
        return pca_final_score > k_final_score
    def choose_best_pca(self, data, pca_params):
        # Check if PCA parameters are provided
        if pca_params is None or len(pca_params) == 0:
            print("PCA parameters not provided. Skipping PCA.")
            return

        # Rest of your code for PCA computations
        best_pca_model = None
        best_scaler = None
        best_normalizer = None

        # Separate numeric and non-numeric data
        numeric_data = data[self.numeric_columns]
        non_numeric_data = data.drop(columns=self.numeric_columns)
        scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
        normalizers = [Normalizer(), Binarizer(), MinMaxScaler()]
        print('Peforming feature selection on just categorical')
        y = self.response_data  
        list_of_k = self.k_to_consider
        classification = self.classification 
        regression = self.regression
        best_k, best_features = self.select_best_k(X = non_numeric_data, y= y, list_of_k = list_of_k, classification = classification, regression = regression)
        print('Saving categorical columns to keep')
        current_non_numeric_data = non_numeric_data.iloc[:, best_features]
        self.cat_columns_to_keep = current_non_numeric_data.columns
        for scaler in scalers:
            for normalizer in normalizers:
                print(f'scaling with {scaler}')
                scaled_data = scaler.fit_transform(numeric_data) if scaler else numeric_data
                normalized_data = normalizer.fit_transform(scaled_data) if normalizer else scaled_data
                print(f'normalizing with {normalizer}')
                # Concatenate normalized numeric data with non-numeric data
                for component in self.PCA:
                    transformed_data = pd.concat([pd.DataFrame(normalized_data, index=numeric_data.index, columns = numeric_data.columns), current_non_numeric_data], axis=1)
                    print(f'Performing PCA with {component} components')
                    pca = PCA(n_components=component)
                    pca_transformed_data = pca.fit_transform(transformed_data[numeric_data.columns])
                    pca_df = pd.DataFrame(pca_transformed_data)
                    transformed_data = pd.concat([pd.DataFrame(pca_transformed_data, index=numeric_data.index), current_non_numeric_data], axis=1)
                    self.score_dict[f'PCA_{component}_{scaler}_{normalizer}']= self.perform_cross_validation(pca_transformed_data, response_variable,   self.classification, self.regression)    
                    score =  self.score_dict[f'PCA_{component}_{scaler}_{normalizer}']
                    if self.compare_scores(score, self.best_pca_scoring):
                        self.best_pca_scoring = score
                        self.best_pca_model = pca
                        self.best_pca_scaler = scaler
                        self.best_pca_normalizer = normalizer   
    def perform_cross_validation(self, X, y, classification, regression):
        np.random.seed(42)
        scores = []
        if classification:
            model = LogisticRegression(random_state=self.random_state)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scoring_function = accuracy_score
        elif regression:
            model = LinearRegression(random_state=self.random_state)
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scoring_function = r2_score
        else:
            raise ValueError("Either classification or regression must be True.")

        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            score = scoring_function(y_test, y_pred)
            scores.append(score)

        return np.mean(scores), np.std(scores)
    def select_best_k(self, X, y, list_of_k, classification, regression, scoring=None):
        np.random.seed(random_state)
        task = 'classification' if classification else 'regression'
        best_score = -np.inf
        best_k = np.inf
        best_features = None
        for k in list_of_k:
            # Feature selection
            selector = SelectKBest(k=k).fit(X, y)
            X_selected = selector.transform(X)

            # Cross-validation
            if classification:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)  # Add random_state
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)  # Add random_state for KFold

            scores = []
            for train_index, test_index in cv.split(X_selected, y):
                X_train, X_test = X_selected[train_index], X_selected[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Model fitting and prediction
                if classification:
                    model = LogisticRegression(random_state=self.random_state)
                else:
                    model = LinearRegression()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                score = r2_score(y_test, y_pred) if regression else accuracy_score(y_test, y_pred)
                scores.append(score)

            mean_score = np.mean(scores)
            std_score = np.std(scores)

            # Adjust these weights as needed
            weight_for_mean = 1.0
            weight_for_std = -0.5  # Lower std is better
            final_score = mean_score * weight_for_mean + std_score * weight_for_std

            # Compare and store the best score and corresponding k
            if final_score > best_score:
                best_score = final_score
                best_k = k
                best_features = selector.get_support(indices=True)

        return best_k, best_features

        return best_k, best_features
    def determine_best_k_for_feature_selection(self, numeric_features, categorical_features, k_params, response_var):
        np.random.seed(self.random_state)
        scalers = [StandardScaler(), RobustScaler(), None]
        normalizers = [MinMaxScaler(), None]
        best_k = np.inf
        for scaler in scalers:
            for normalizer in normalizers:
                scaled_data = scaler.fit_transform(numeric_features) if scaler else numeric_features
                normalized_data = normalizer.fit_transform(scaled_data) if normalizer else scaled_data
                transformed_data = pd.concat([pd.DataFrame(normalized_data, index=numeric_features.index, columns=numeric_features.columns), categorical_features], axis=1)
                for k in k_params:
                    print(f'Performing select {k} best with {scaler} scaler and {normalizer} normalizer')
                    selector = SelectKBest(k=k).fit(transformed_data, response_var)
                    X_selected = selector.transform(transformed_data)
                    selected_mask = selector.get_support()  # Boolean mask of selected features
                    original_columns = transformed_data.columns  # Get the original column names
                    selected_column_names = original_columns[selected_mask]
                    # Separate the response variable from the transformed data
                    y = response_var
                    # Perform cross-validation
                    self.score_dict[f'Kbest_{k}_{scaler}_{normalizer}'] = self.perform_cross_validation(X_selected, y, self.classification, self.regression)  
                    score = self.score_dict[f'Kbest_{k}_{scaler}_{normalizer}']
                    if self.compare_scores_for_k(score, self.best_k_scoring, k, best_k):
                        self.best_k_scoring = score
                        self.total_columns_to_keep = selected_column_names
                        self.k_scaler = scaler
                        self.k_normalizer = normalizer  
                        best_k = k

        print(f'Best Select K is {best_k} with {self.k_scaler} scaler and {self.k_normalizer}')

    def compare_scores_for_k(self, pca_score, k_score, k_pca, k_kbest):
        pca_mean, pca_std = pca_score
        k_mean, k_std = k_score

        # Adjust these weights as needed
        weight_for_mean = 1.0
        weight_for_std = -0.05  # Negative since lower std is better
        regularization_strength = 0.001  # Adjust this to control the regularization effect

        # Regularize the scores by subtracting a small value proportional to k
        pca_final_score = (pca_mean * weight_for_mean + pca_std * weight_for_std) - (regularization_strength * k_pca)
        k_final_score = (k_mean * weight_for_mean + k_std * weight_for_std) - (regularization_strength * k_kbest)

        return pca_final_score > k_final_score
    def prepare_data_for_phase_4(self, data, response_var, numeric_columns, scaler, normalizer, pca_model, selected_features, file_output = None):
        """
        Prepare data by applying scaling, normalization, and PCA (if applicable).
        Also, selects the appropriate features for the final dataset.

        Parameters:
        - data: DataFrame, the original dataframe
        - scaler: Scaler object for scaling (e.g., StandardScaler)
        - normalizer: Normalizer object for normalization (e.g., MinMaxScaler)
        - pca_model: PCA model object if PCA was chosen, otherwise None
        - selected_features: list of column names to keep after feature selection

        Returns:
        - final_data: DataFrame, the processed data
        """
        response_data = data.pop(response_var)
        numeric_data = data[numeric_columns]
        categorical_data = data.drop(columns=numeric_columns)

        # Apply scaling and normalization to numeric data
        scaled_numeric = scaler.transform(numeric_data) if scaler else numeric_data
        normalized_numeric = normalizer.transform(scaled_numeric) if normalizer else scaled_numeric

        if pca_model:
            transformed_numeric = pca_model.transform(normalized_numeric)
            numeric_df = pd.DataFrame(transformed_numeric, index=numeric_data.index)
            if selected_features:
                final_data = pd.concat([numeric_df, categorical_data[selected_features]], axis=1)
            else:
                final_data = pd.concat([numeric_df, categorical_data], axis=1)
        else:
            numeric_df = pd.DataFrame(normalized_numeric, index=numeric_data.index, columns=numeric_data.columns)
            final_data = pd.concat([numeric_df, categorical_data], axis=1)
            final_data = final_data[selected_features]
       
     

        final_data = pd.concat([final_data, response_data], axis=1)
        if file_output is not None:
            print(f'Sending data to {file_output}')
            final_data.to_csv(file_output, index = True)

        return final_data
def prepare_test_data_for_phase_4(data, response_var, numeric_columns, scaler, normalizer, pca_model, selected_features, file_output = None):
        """
        Prepare data by applying scaling, normalization, and PCA (if applicable).
        Also, selects the appropriate features for the final dataset.

        Parameters:
        - data: DataFrame, the original dataframe
        - scaler: Scaler object for scaling (e.g., StandardScaler)
        - normalizer: Normalizer object for normalization (e.g., MinMaxScaler)
        - pca_model: PCA model object if PCA was chosen, otherwise None
        - selected_features: list of column names to keep after feature selection

        Returns:
        - final_data: DataFrame, the processed data
        """
        response_data = data.pop(response_var)
        numeric_data = data[numeric_columns]
        categorical_data = data.drop(columns=numeric_columns)

        # Apply scaling and normalization to numeric data
        scaled_numeric = scaler.transform(numeric_data) if scaler else numeric_data
        normalized_numeric = normalizer.transform(scaled_numeric) if normalizer else scaled_numeric

        if pca_model:
            transformed_numeric = pca_model.transform(normalized_numeric)
            numeric_df = pd.DataFrame(transformed_numeric, index=numeric_data.index)
            if selected_features:
                final_data = pd.concat([numeric_df, categorical_data[selected_features]], axis=1)
            else:
                final_data = pd.concat([numeric_df, categorical_data], axis=1)
        else:
            numeric_df = pd.DataFrame(normalized_numeric, index=numeric_data.index, columns=numeric_data.columns)
            final_data = pd.concat([numeric_df, categorical_data], axis=1)
            final_data = final_data[selected_features]
       
     

        final_data = pd.concat([final_data, response_data], axis=1)
        if file_output is not None:
            print(f'Sending data to {file_output}')
            final_data.to_csv(file_output, index = True)

        return final_data


class Phase4(BaseEstimator, TransformerMixin):
    def __init__(self, response_variable, classifier_param_tuples, binary=True, significance_level=0.05, random_state=42, output_path = None, dict_output = None, model_output_path = None):
        self.classifier_param_tuples = classifier_param_tuples
        self.binary = binary
        self.significance_level = significance_level
        self.random_state = random_state
        self.best_classifier = None
        self.best_classifier_name = ''
        self.best_score = 0
        self.best_params = {}
        self.performance_grid = {}
        self.baseline_name = 'LogisticRegression' if self.binary else 'LinearRegression'
        self.response_variable = response_variable
        self.output_path = output_path
        self.dict_output = dict_output
        self.model_output_path = model_output_path

    def fit(self, X, y=None):
        y_train = X.pop(self.response_variable)
        X_train = X
        self.determine_baseline(X_train, y_train)
        any_classifier_significantly_better = False

        for classifier, param_grid in self.classifier_param_tuples:
            classifier_name = classifier.__class__.__name__
            start_time = time.time()

            # Ensure the classifier has the correct random_state if applicable
            if hasattr(classifier, 'random_state'):
                classifier.set_params(random_state=self.random_state)

            param_combinations = self._param_grid_to_param_combinations(param_grid)
            best_score = 0
            best_params = None

            for params in param_combinations:
                classifier.set_params(**params)
                scores = self.perform_cross_validation(classifier, X_train, y_train)
                mean_score = np.mean(scores)
                std_score = np.std(scores)

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params

            training_time = time.time() - start_time

            self.performance_grid[classifier_name] = {
                'mean_accuracy': best_score,
                'std_accuracy': std_score,
                'training_time': training_time,
                'best_params': best_params
            }
            print(self.performance_grid[classifier_name])

            is_better_than_baseline = self.is_significantly_better(best_score, std_score, classifier_name)
            if is_better_than_baseline:
                any_classifier_significantly_better = True
                print(f'{classifier_name} outperforms baseline')
                if best_score > self.best_score:
                    self.best_classifier = classifier
                    self.best_classifier_name = classifier_name
                    self.best_score = best_score
                    self.best_params = best_params

                    # Fit the best classifier on the training data
                    self.best_classifier.set_params(**self.best_params)
                    self.best_classifier.fit(X_train, y_train)
                    print(f"Best classifier {self.best_classifier_name} is fitted with best parameters.")
            else:
                print(f'{classifier_name} does not outperform baseline')

        # If no classifier is significantly better than the baseline, use the baseline
        if not any_classifier_significantly_better:
            self.best_classifier = LogisticRegression() if self.binary else LinearRegression()
            self.best_classifier_name = self.baseline_name
            self.best_score = self.performance_grid[self.baseline_name]['mean_accuracy']
            self.best_params = self.performance_grid[self.baseline_name]['best_params']

            # Fit the baseline classifier
            self.best_classifier.set_params(**self.best_params)
            self.best_classifier.fit(X_train, y_train)
            print(f"Baseline classifier {self.best_classifier_name} is fitted as the best classifier.")
        if self.dict_output is not None:
            print(f'Sending score_dict to {self.dict_output}')
            with open(self.dict_output, 'w') as file:
                json.dump(self.performance_grid, file, indent=4)

        return self

    def _param_grid_to_param_combinations(self, param_grid):
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return param_combinations

    def transform(self, X=None):
        if self.model_output_path is not None:
            with open(self.model_output_path, 'wb') as model_file:
                print(f'Sending pickle file of model to {self.model_output_path}')
                pickle.dump(self.best_classifier, model_file)  # Corrected here
        else:
            print("No model_output_path provided, skipping model save.")
        return self.best_classifier, self.best_score, self.performance_grid

    def determine_baseline(self, X_train, y_train):
        baseline_model = LogisticRegression() if self.binary else LinearRegression()
        baseline_model.fit(X_train, y_train)
        cv_scores = self.perform_cross_validation(baseline_model, X_train, y_train)
        mean_accuracy = np.mean(cv_scores)
        std_accuracy = np.std(cv_scores)
        self.performance_grid[self.baseline_name] = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'training_time': None,
            'best_params': baseline_model.get_params()
        }

    def is_significantly_better(self, mean_accuracy, std_accuracy, classifier_name):
        baseline_mean_accuracy = self.performance_grid[self.baseline_name]['mean_accuracy']
        baseline_std_accuracy = self.performance_grid[self.baseline_name]['std_accuracy']

        # Calculate the difference in means and its standard error
        mean_diff = mean_accuracy - baseline_mean_accuracy
        print(f'Mean score is {mean_accuracy} and baseline is {baseline_mean_accuracy}')
        std_error_diff = np.sqrt((std_accuracy**2) / 5 + (baseline_std_accuracy**2) / 5)

        # Calculate the 95% confidence interval of the difference
        confidence_interval = [mean_diff - 1.96*std_error_diff, mean_diff + 1.96*std_error_diff]
        print(f'Confidence interval for difference between  mean score and baseline is {confidence_interval}')

        # Check if the confidence interval is entirely above zero
        return confidence_interval[0] > 0

    def perform_cross_validation(self, classifier, X_train, y_train):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring=make_scorer(accuracy_score))
        return scores
def predict_with_dropped_indices(test_data, model, original_data_path, dropped_indices, file_output=None, random_state=42):
    np.random.seed(random_state)
    # Load the original dataset
    original_data = pd.read_csv(original_data_path, index_col=0)

    # Ensure the test data does not contain the response variable
    if 'make_mlb' in test_data:
        test_data = test_data.drop(columns=['make_mlb'])

    # Filter the original data to include only the dropped indices
    original_dropped_data = original_data[original_data['key_bbref_minors'].isin(dropped_indices)]
    
    # Predict probabilities for the test data points not in dropped_indices
    probabilities = model.predict_proba(test_data)[:, 1]  # get the probability for the positive class
    
    # Create a DataFrame from the predictions
    predictions_df = pd.DataFrame({
        'key_bbref_minors': test_data.index.astype(str),  # Convert index to string
        'predicted_probability': probabilities
    })

    # Convert 'key_bbref_minors' in original_data to string
    original_data['key_bbref_minors'] = original_data['key_bbref_minors'].astype(str)

    # Merge with the original dataset to add the make_mlb values
    merged = predictions_df.merge(original_data[['key_bbref_minors', 'make_mlb']], on='key_bbref_minors', how='left')
    
    # Create a DataFrame for dropped indices with their actual make_mlb values
    dropped_df = pd.DataFrame({
        'key_bbref_minors': original_dropped_data['key_bbref_minors'],
        'predicted_probability': ["can't be predicted"] * len(original_dropped_data),
        'make_mlb': original_dropped_data['make_mlb']
    })
    
    # Concatenate the results with the dropped indices DataFrame
    final_output = pd.concat([merged, dropped_df], ignore_index=True)
    if file_output is not None:
        print(f'Sending data to {file_output}')
        final_output.to_csv(file_output, index=True)
    
    return final_output
def find_best_threshold(df, response_col, prob_col, thresholds, random_state = 42):
    np.random.seed(random_state)
    best_threshold = 0
    best_gmean = 0
    best_sensitivity = 0
    best_specificity = 0

    # Ensure the predicted_probability column is numeric
    if not np.issubdtype(df[prob_col].dtype, np.number):
        df[prob_col] = pd.to_numeric(df[prob_col], errors='coerce')

    for threshold in thresholds:
        # Convert probabilities to binary predictions based on the threshold
        predictions = (df[prob_col] >= threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(df[response_col], predictions).ravel()
        
        # Calculate sensitivity and specificity
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        
        # Calculate geometric mean of sensitivity and specificity
        gmean = np.sqrt(sensitivity * specificity)

        # Update best threshold, gmean, sensitivity, and specificity
        if gmean > best_gmean:
            best_gmean = gmean
            best_threshold = threshold
            best_sensitivity = sensitivity
            best_specificity = specificity

    # Interpretation
    print(f"Best Threshold: {best_threshold}")
    print(f"Geometric Mean of Sensitivity and Specificity: {best_gmean:.2f}")
    print(f"Sensitivity (True Positive Rate): {best_sensitivity:.2f} - This model correctly predicts {best_sensitivity * 100:.2f}% of those who made it to the MLB.")
    print(f"Specificity (True Negative Rate): {best_specificity:.2f} - This model correctly predicts {best_specificity * 100:.2f}% of those who did not make it to the MLB.")

    return best_threshold, best_gmean, best_sensitivity, best_specificity

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def perform_ablation_test_classification(training_data_for_model, test_data_for_model, predictor, response_var, original_data_path, original_geo_mean, dropped_indices, col_to_consider, file_output=None, classification=True, regression=False, random_state=42):
    np.random.seed(random_state)
    thresholds = np.arange(0.1, 1, 0.01)
    performance_without_variables = pd.DataFrame(columns=['Variable', 'Original Geo Mean', 'New Geomean', 'Geomean Change'])
    
    for variable in col_to_consider:
        print(f'Dropping {variable} from both training and test set')
        
        # Exclude the response variable from being dropped
        if variable == response_var:
            continue
        
        train_data = training_data_for_model.drop(columns=variable, errors='ignore')
        test_data = test_data_for_model.drop(columns=variable, errors='ignore')
        
        # Confirm that the response variable is present before proceeding
        if response_var not in train_data.columns:
            raise KeyError(f"Response variable '{response_var}' not found in the training data.")
        
        train_y = train_data.pop(response_var)

        # Ensure consistent data types for merging (if applicable)
        # Assuming 'predict_with_dropped_indices' merges datasets
        # Modify or add type conversion as needed based on the actual merging logic
        for column in train_data.columns:
            if train_data[column].dtype != test_data[column].dtype:
                train_data[column] = train_data[column].astype('float')
                test_data[column] = test_data[column].astype('float')

        print(f'Fitting {predictor} to train data')

        # Fit the predictor to the training data
        if hasattr(predictor, 'random_state'):
            predictor.set_params(random_state=random_state)
        predictor.fit(train_data, train_y)

        final_prediction_test_set = predict_with_dropped_indices(test_data, predictor, original_data_path, dropped_indices, file_output=None)
        best_threshold, best_gmean, best_sensitivity, best_specificity = find_best_threshold(final_prediction_test_set, 'make_mlb', 'predicted_probability', thresholds)
        geometric_mean_change = original_geo_mean - best_gmean
        
        performance_without_variables = performance_without_variables.append({
            'Variable': variable,
            'Original Geo Mean': round(original_geo_mean, 3),
            'New Geomean': round(best_gmean, 3),
            'Geomean Change': round(geometric_mean_change, 3)
        }, ignore_index=True)
    
    if file_output is not None:
        performance_without_variables.to_csv(file_output, index=True)
    
    return performance_without_variables.sort_values(by='Geomean Change', ascending=True)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

import numpy as np
import pandas as pd
import pickle
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

def calculate_batch_size(samples_needed):
    min_batch_size = 50
    max_batch_size = 500
    batch_size = int(samples_needed * 0.1)
    batch_size = max(min_batch_size, min(batch_size, max_batch_size))
    batch_size = batch_size - (batch_size % 10)
    return batch_size
def set_hyperparameters(df):
    # More epochs for larger datasets
    epochs = 500 if len(df) > 10000 else 300

    # Adjust dimensions based on the number of features
    if len(df.columns) > 50:
        generator_dim = discriminator_dim = (256, 256, 256)
    else:
        generator_dim = discriminator_dim = (128, 128, 128)

    # Learning rates
    generator_lr = discriminator_lr = 2e-4

    return epochs, generator_dim, discriminator_dim, generator_lr, discriminator_lr

def balance_and_generate_synthetic_data(input_path, response_var, output_path, model_output_path, per, random_state=42, drop_columns=None, drop_columns_that_begin_with=None, col_to_keep=None):
    np.random.seed(random_state)
    df = pd.read_csv(input_path, index_col=0)

    if drop_columns:
        df.drop(columns=drop_columns, errors='ignore', inplace=True)
    if drop_columns_that_begin_with:
        df = df[[col for col in df.columns if not col.startswith(drop_columns_that_begin_with) or col == col_to_keep]]
        print(f'Dropped some columns for computational efficiency. Now left with {len(df.columns)} columns.')

    class_counts = df[response_var].value_counts()
    minority_class_count = class_counts.min()
    majority_class_count = class_counts.max()
    print(f'We have {minority_class_count} in the minority class and {majority_class_count} in the majority class')
    difference_needed = majority_class_count - minority_class_count
    samples_needed = int(difference_needed * (1 + per))
    
    print(f'Trying {samples_needed} synthetic samples to generate at least {difference_needed} synthetic data points with the minority label.')

    batch_size = calculate_batch_size(samples_needed)
    epochs, generator_dim, discriminator_dim, generator_lr, discriminator_lr = set_hyperparameters(df)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    synthesizer = CTGANSynthesizer(metadata, epochs=epochs, batch_size=batch_size, generator_dim=generator_dim, discriminator_dim=discriminator_dim, generator_lr=generator_lr, discriminator_lr=discriminator_lr, verbose=False)
    print('Fitting synthesizer')

    synthesizer.fit(df)
    print('Generating synthetic data')

    synthetic_data = synthesizer.sample(num_rows=int(samples_needed))
    synthetic_ones = synthetic_data[synthetic_data[response_var] == 1]
    print(f'We have {len(synthetic_ones)} data points in the minority class')
    # Sample only the needed number of synthetic '1's
    if len(synthetic_ones) > difference_needed:
        print(f'Random Sampling {difference_needed} from our {len(synthetic_ones)} aso we have class balance')
        synthetic_ones_sampled = synthetic_ones.sample(n=difference_needed)
    elif len(synthetic_ones) < difference_needed:
        print('Need to generate more synthetic data.')
        return balance_and_generate_synthetic_data(input_path, response_var, output_path, model_output_path, per=current_per + 0.05, drop_columns_that_begin_with = drop_columns_that_begin_with, col_to_keep = col_to_keep)
    else:
        synthetic_ones_sampled = synthetic_ones

    # Assign unique identifiers to synthetic data
    print('Assinging identifiers for synthetic data points')
    synthetic_ones_sampled['synthetic_id'] = ['synth_' + str(i) for i in range(1, len(synthetic_ones_sampled) + 1)]
    synthetic_ones_sampled.set_index('synthetic_id', inplace=True)

    balanced_synthetic_data = pd.concat([df, synthetic_ones_sampled])
    print(f'Sending csv file to {output_path}')
    balanced_synthetic_data.to_csv(output_path)

    if model_output_path is not None:
        with open(model_output_path, 'wb') as model_file:
            print(f'Sending model file to {model_file}')
            pickle.dump(synthesizer, model_file)

    return balanced_synthetic_data, synthesizer



def visualize_misclassifications(phase_3_test_data, phase_2_test_data, predictions, response_col, prob_col, threshold, key, pdf_output):
    # Merge the test data with the predictions on the key column
    merged_df = pd.merge(predictions[[prob_col, key]], phase_3_test_data, on=key)

    # Calculate predicted class
    merged_df['predicted'] = (merged_df[prob_col] >= threshold).astype(int)
    merged_df['actual'] = merged_df[response_col]

    # Create a subset DataFrame for misclassified observations
    misclassified_df = merged_df[merged_df['predicted'] != merged_df['actual']]

    # Select all columns from phase_3_test_data except for the response variable
    columns = [col for col in phase_3_test_data.columns if col != response_col]

    # Merge with phase_2_test_data
    final_df = pd.merge(misclassified_df[[prob_col, key, 'predicted', 'actual']], phase_2_test_data[columns], on=key)

    # Identify binary variables
    binary_vars = [col for col in final_df.columns if final_df[col].nunique() == 2 and col != 'predicted' and col != 'actual']

    # Identify numeric variables
    numeric_vars = final_df.select_dtypes(include='number').columns.tolist()
    numeric_vars = [var for var in numeric_vars if var not in {'predicted', 'actual', prob_col} and var not in binary_vars]
    total_vars = len(binary_vars) + len(numeric_vars)
    cols = 2  # Assuming 2 columns for the layout
    rows = total_vars // cols + (total_vars % cols > 0)

    # Create a PDF file to save the plots
    with PdfPages(pdf_output) as pdf:
        plt.figure(figsize=(10 * cols, 6 * rows))

        # Plot for each binary variable
        for i, var in enumerate(binary_vars):
            plt.subplot(rows, cols, i + 1)
            sns.boxplot(x=var, y=prob_col, data=final_df)
            plt.title(f'Predicted Probability Distribution for {var}')

        # Plot for each numeric variable with larger, more distinct dots
        for i, var in enumerate(numeric_vars, start=len(binary_vars)):
            plt.subplot(rows, cols, i + 1)
            sns.scatterplot(
                x=var, y=prob_col, data=final_df,
                hue='actual', palette='bright',  # Use a bright palette for distinct colors
                s=100  # Set the size of the dots to be larger, e.g., 100
            )
            plt.title(f'Predicted Probability vs {var}')
            plt.legend(title='Made MLB', loc='best')

        plt.tight_layout()  # Adjust the layout
        pdf.savefig()  # saves the current figure
        plt.close()    # close the figure window

    return misclassified_df
def refit_model_and_get_new_misclassified(columns_to_drop, predictor, response_col, prob_col, pdf_output,  file_output, phase_2_test_data, dropped_indices, phase_3_training_data, phase_3_test_data, thresholds, original_data_path, key):
    Test = phase_3_test_data.copy()
    train_data = phase_3_training_data.drop(columns=columns_to_drop, errors='ignore')
    test_data = phase_3_test_data.drop(columns=columns_to_drop, errors='ignore')
    train_y = train_data.pop(response_col)
    
    print(f'Fitting {predictor} to train data without {columns_to_drop}')
    predictor.fit(train_data, train_y)
    final_prediction_test_set = predict_with_dropped_indices(test_data, predictor, original_data_path, dropped_indices, file_output= file_output)
    best_threshold, best_gmean, best_sensitivity, best_specificity = find_best_threshold(final_prediction_test_set, 'make_mlb', 'predicted_probability', thresholds)
    new_misclassified_df = visualize_misclassifications(Test, phase_2_test_data, predictions = final_prediction_test_set, response_col = response_col , prob_col= prob_col, key = key, pdf_output = pdf_output, threshold = best_threshold)
    
    print('Returning new misclassifed df, new predicted set, new   best_threshold, best_gmean, best_sensitivity, best_specificity, new predictor')
    
    return new_misclassified_df, final_prediction_test_set, best_threshold, best_gmean, best_sensitivity, best_specificity, predictor





import matplotlib.backends.backend_pdf
def boxplots_for_numerics_to_pdf(df,columns, pdf_filename):
    # Identify numeric columns in the dataframe
    df = df[columns]
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    # Filter columns that have more than 2 unique values
    numeric_columns = [col for col in numeric_columns if df[col].nunique() > 2]

    # Create a PDF file to save plots
    pdf_pages = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)

    for variable in numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[variable])
        plt.title(f'Boxplot of {variable} for My Training Set', fontsize=16)
        plt.xlabel('Training Set', fontsize=14)
        plt.ylabel(variable, fontsize=14)
        plt.grid(True)

        # Save the current figure to the pdf file
        pdf_pages.savefig(plt.gcf())
        plt.close()

    # Close the PDF file
    pdf_pages.close()
    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def plot_data_with_synthetic_to_pdf(phase_2_df, phase_3_df, response_col, pdf_output_path):
    # Separate synthetic and regular data
    synthetic_data = phase_2_df[phase_2_df.index.str.startswith('synth')]
    regular_data = phase_2_df[~phase_2_df.index.str.startswith('synth')]

    # Identify numeric and categorical columns
    numeric_cols = phase_3_df.select_dtypes(include='number').columns.tolist()
    # Identify categorical columns, excluding the numeric columns and the response column
    categorical_cols = [col for col in phase_3_df.columns if phase_3_df[col].nunique() <= 2 
                        and col not in numeric_cols and col != response_col]

    with PdfPages(pdf_output_path) as pdf:
        # Box plots for numeric columns
        for col in numeric_cols:
            if col != response_col:
                fig, axes = plt.subplots(1, 2, figsize=(18, 6))

                # Boxplot for regular data
                sns.boxplot(ax=axes[0], x=regular_data[response_col], y=regular_data[col])
                axes[0].set_title(f'Regular Data: {col}')
                axes[0].set_xlabel(response_col)
                axes[0].set_ylabel(col)

                # Boxplot for synthetic data (all have the same make_mlb value)
                sns.boxplot(ax=axes[1], y=synthetic_data[col], whis=[0, 100])
                axes[1].set_title(f'Synthetic Data: {col}')
                axes[1].set_ylabel(col)
                axes[1].set_xlabel('Synthetic')

                pdf.savefig(fig)  # Save the figure to the PDF
                plt.close(fig)

        # Tables for categorical columns
        for col in categorical_cols:
            value_counts_regular = regular_data[col].value_counts(normalize=True)
            value_counts_synthetic = synthetic_data[col].value_counts(normalize=True)

            cross_tab_regular = pd.crosstab(regular_data[col], regular_data[response_col], margins=True)
            cross_tab_synthetic = pd.crosstab(synthetic_data[col], synthetic_data[response_col], margins=True)
            
            # Save the value counts table to the PDF
            fig, ax = plt.subplots(figsize=(12, 2))
            ax.axis('tight')
            ax.axis('off')
            ax.table(cellText=[value_counts_regular.values, value_counts_synthetic.values],
                     rowLabels=['Regular', 'Synthetic'],
                     colLabels=value_counts_regular.index,
                     cellLoc = 'center', rowLoc = 'center',
                     loc='center')
            ax.set_title(f'Value Counts for {col}')
            pdf.savefig(fig)
            plt.close(fig)

            # Save the crosstab table to the PDF
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.axis('tight')
            ax.axis('off')
            ax.table(cellText=cross_tab_regular.values, colLabels=cross_tab_regular.columns,
                     rowLabels=cross_tab_regular.index,
                     cellLoc = 'center', rowLoc = 'center',
                     loc='center')
            ax.set_title(f'Cross-tabulation for {col} - Regular Data')
            pdf.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.axis('tight')
            ax.axis('off')
            ax.table(cellText=cross_tab_synthetic.values, colLabels=cross_tab_synthetic.columns,
                     rowLabels=cross_tab_synthetic.index,
                     cellLoc = 'center', rowLoc = 'center',
                     loc='center')
            ax.set_title(f'Cross-tabulation for {col} - Synthetic Data')
            pdf.savefig(fig)
            plt.close(fig)
def create_boxplots_and_save_to_pdf(phase_2_data, phase_3_data, pdf_path):
    # Identify numeric columns with more than one unique value in both datasets
    numeric_cols_phase_2 = phase_2_data.select_dtypes(include=np.number).columns
    numeric_cols_phase_3 = phase_3_data.select_dtypes(include=np.number).columns
    common_numeric_cols = [col for col in numeric_cols_phase_2 if col in numeric_cols_phase_3 and
                           phase_2_data[col].nunique() > 2 and phase_3_data[col].nunique() > 2]
    
    # Create a PDF to save the boxplots
    with PdfPages(pdf_path) as pdf:
        for col in common_numeric_cols:
            print(f'Creating BoxPlot for {col}')
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns for boxplots
            
            # Boxplot for phase 2 data
            sns.boxplot(ax=axes[0], y=phase_2_data[col])
            axes[0].set_title(f'{col} - Before Scaling and Normalizing')
            axes[0].set_ylabel('Values')
            axes[0].set_xlabel('Phase 2 Data')

            # Boxplot for phase 3 data
            sns.boxplot(ax=axes[1], y=phase_3_data[col])
            axes[1].set_title(f'{col} - After Scaling and Normalizing')
            axes[1].set_ylabel('Values')
            axes[1].set_xlabel('Phase 3 Data')
            
            plt.tight_layout()
            pdf.savefig(fig)  # Save the current figure to pdf
            plt.close(fig)  # Close the figure to avoid display
    print(f'Sent to {pdf_path}')


