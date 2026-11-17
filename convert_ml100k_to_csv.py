#!/usr/bin/env python3
"""
Made by AI
Convert the old MovieLens 100K format (ml-100k) into the modern MovieLens CSV
format used by ml-latest-small, ml-25m, ml-32m, etc.

Input  (ml-100k):                      Output (modern format):
  u.data  (tab-separated, no header)     ratings.csv  userId,movieId,rating,timestamp
  u.item  (pipe-separated, latin-1)      movies.csv   movieId,title,genres

Differences handled:
  - u.data has no header and is tab-separated  -> ratings.csv with a header, commas
  - ratings are integers (1-5)                 -> written as floats (4 -> 4.0)
  - u.item stores genres as 19 0/1 flag columns -> one pipe-joined genres string
        e.g. "Animation|Children|Comedy"  ("Children's" is renamed to "Children",
        the "unknown" flag is dropped, no genres -> "(no genres listed)")
  - u.item is latin-1 encoded                  -> output files are UTF-8

Not produced (ml-100k has no equivalent data): tags.csv and links.csv.
The u.user demographics file is not part of the modern format and is ignored.

Usage:
  python convert_ml100k_to_csv.py
  python convert_ml100k_to_csv.py --input data/ml-100k --output data/ml-100k-csv
  python convert_ml100k_to_csv.py --download      # fetch ml-100k first if missing
"""

import argparse
import os
import sys
import urllib.request
import zipfile

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'

# Order of the 19 genre flag columns in u.item
GENRE_FLAGS = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
]
GENRE_RENAME = {"Children's": 'Children'}
NO_GENRES = '(no genres listed)'


def download_ml100k(input_dir):
    parent = os.path.dirname(os.path.abspath(input_dir))
    os.makedirs(parent, exist_ok=True)
    zip_path = os.path.join(parent, 'ml-100k.zip')
    print(f'Downloading {URL} ...')
    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(parent)
    print(f'Extracted to {parent}')


def convert_ratings(input_dir):
    path = os.path.join(input_dir, 'u.data')
    df = pd.read_csv(
        path, sep='\t', header=None,
        names=['userId', 'movieId', 'rating', 'timestamp'],
    )
    df['rating'] = df['rating'].astype(float)
    df = df.sort_values(['userId', 'timestamp', 'movieId'], kind='stable').reset_index(drop=True)
    return df


def convert_movies(input_dir):
    path = os.path.join(input_dir, 'u.item')
    cols = ['movieId', 'title', 'release_date', 'video_release_date', 'imdb_url'] + GENRE_FLAGS
    raw = pd.read_csv(path, sep='|', header=None, names=cols, encoding='latin-1')

    def genres_of(row):
        names = [GENRE_RENAME.get(g, g) for g in GENRE_FLAGS[1:] if row[g] == 1]
        return '|'.join(names) if names else NO_GENRES

    out = pd.DataFrame({
        'movieId': raw['movieId'],
        'title': raw['title'].str.strip(),
        'genres': raw.apply(genres_of, axis=1),
    })
    return out.sort_values('movieId').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description='Convert ml-100k to the modern MovieLens CSV format.')
    parser.add_argument('--input', default=os.path.join(SCRIPT_DIR, 'data', 'ml-100k'),
                        help='folder containing u.data and u.item (default: data/ml-100k next to this script)')
    parser.add_argument('--output', default=os.path.join(SCRIPT_DIR, 'data', 'ml-100k-csv'),
                        help='folder to write ratings.csv and movies.csv (default: data/ml-100k-csv)')
    parser.add_argument('--download', action='store_true',
                        help='download and extract ml-100k if the input folder does not exist')
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        if args.download:
            download_ml100k(args.input)
        else:
            sys.exit(f'Input folder not found: {args.input}\n'
                     f'Pass --input <folder with u.data and u.item>, or use --download.')

    for name in ('u.data', 'u.item'):
        if not os.path.isfile(os.path.join(args.input, name)):
            sys.exit(f'Missing {name} in {args.input}')

    ratings = convert_ratings(args.input)
    movies = convert_movies(args.input)

    # Sanity checks
    missing = set(ratings['movieId']) - set(movies['movieId'])
    if missing:
        print(f'WARNING: {len(missing)} movieIds in ratings are not in movies (e.g. {sorted(missing)[:5]})')
    if ratings.duplicated(['userId', 'movieId']).any():
        print('WARNING: duplicate (userId, movieId) pairs found in ratings')

    os.makedirs(args.output, exist_ok=True)
    ratings_path = os.path.join(args.output, 'ratings.csv')
    movies_path = os.path.join(args.output, 'movies.csv')
    ratings.to_csv(ratings_path, index=False, encoding='utf-8')
    movies.to_csv(movies_path, index=False, encoding='utf-8')

    print(f'Wrote {ratings_path}  ({len(ratings):,} ratings, '
          f'{ratings.userId.nunique():,} users, {ratings.movieId.nunique():,} rated movies)')
    print(f'Wrote {movies_path}   ({len(movies):,} movies)')


if __name__ == '__main__':
    main()
