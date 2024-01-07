"""

    This code serves as a data pipeline for the ML NLP model.
    
"""


# import libraries
import pandas as pd
import sys
from sqlalchemy import create_engine


def extract_data(messages_path, categories_path):
    
    messages = pd.read_csv(messages_path)
    categories = pd.read_csv(categories_path)
    
    return messages, categories


def transform_data(messages, categories):
    
    # merge datasets
    df = messages.merge(categories, on='id')

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # fetch categories' names from the first row of data (it looks like: '[category_name]-[bool]')
    row = categories.iloc[0,].str.split('-').apply(lambda x: x[0])
    category_colnames = row.values
    categories.columns = category_colnames


    # replace categories' dummy string values (e.g.: '[category_name]-[bool]') with boolean values as int
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop(columns=['categories'], inplace=True)


    # concatenate messages w/ category features
    df = pd.concat([df, categories], axis=1)


    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def load_data(df, output_path):

    # Load the data to an SQL database
    try:
        engine = create_engine(f'sqlite:///{output_path}/disaster_messages.db')
        df.to_sql(output_path, engine, index=False)
        
    except Exception as e:
        print(e)
        

def main():
    if len(sys.argv) == 4:
        messages_path, categories_path, output_path = sys.argv[1:]
        
        print('Extracting files...\n')
        messages, categories = extract_data(messages_path, categories_path)
        
        print('Transforming files...\n')
        df = transform_data(messages, categories)

        print('Loading files...\n')
        load_data(df, output_path)
        
        print('All done.')
        
    else:
        print(len(sys.argv))
        print(sys.argv)
        print('This function requires three positional arguments:\n-a path to messages file\n-a path to categories file\n-a path the output should be saved to')
        
    
if __name__ == '__main__':
    main()