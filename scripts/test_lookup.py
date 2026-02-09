from src.service import load_clustered_postcodes, lookup_postcode

df = load_clustered_postcodes()
print(lookup_postcode("EN5 5UJ", df))