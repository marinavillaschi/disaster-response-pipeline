import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Method to load data from both csv files (messages and categories)
    
    Args:
        messages_filepath (str): path to messages .csv file
        categories_filepath (str): path to categories .csv file
        
    Returns:
        dataframe: cancatenated dataframe with messages and categories data
    """
    
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on='id')
    
    return df



def clean_data(df):
    """
    Method to clan data from concatenated datasets
    
    Args:
        df: concatenated dataframe
        
    Returns:
        dataframe: cleaned final dataset
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    
    # extract a list of new column names from the first row and rename the columns with it
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # convert categorie values to just 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
   
    # replace categories column in df with new category columns
    df.drop(columns='categories', inplace=True)
    df_clean = df.merge(categories, left_index=True, right_index=True)
    
    
    # drop duplicates
    df_clean.drop_duplicates(inplace=True)
    
    # drop rows with mistaken values (not 0 and not 1)
    df_clean = df_clean[df_clean.related != 2]
    
    # drop column with only one class (all 0, no 1)
    df_clean.drop(columns='child_alone', inplace = True)
    
    return df_clean
  
    

def save_data(df_clean, database_filename, table_name):
    """
    Method to save dataset into an sqlite database
    
    Args:
        df_clean: clean dataframe
        database_filename (str): name of database to be created
        table_name (str): name of table to create in the database
        
    Returns:
        None
    """
    
    # create engine
    engine = create_engine('sqlite:///'+database_filename)
    
    # load cleaned data into the database, replacing it if it exists already 
    df_clean.to_sql(table_name, engine, index=False, if_exists = 'replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        table_name = 'labeledmessages'

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, table_name)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'disastermanagement.db')


if __name__ == '__main__':
    main()