import sqlite3

import pandas as pd

# read albums table from sqlite
conn = sqlite3.connect("../music.db")

albums = pd.read_sql_query("SELECT * FROM albums", conn)
artists = pd.read_sql_query("SELECT * FROM artists", conn)
customers = pd.read_sql_query("SELECT * FROM customers", conn)
employees = pd.read_sql_query("SELECT * FROM employees", conn)
genres = pd.read_sql_query("SELECT * FROM genres", conn)
invoice_items = pd.read_sql_query("SELECT * FROM invoice_items", conn)
invoices = pd.read_sql_query("SELECT * FROM invoices", conn)
media_types = pd.read_sql_query("SELECT * FROM media_types", conn)
playlist_track = pd.read_sql_query("SELECT * FROM playlist_track", conn)
playlists = pd.read_sql_query("SELECT * FROM playlists", conn)
tracks = pd.read_sql_query("SELECT * FROM tracks", conn)

# How many albums are in the database?
result_1 = albums.shape[0]

# How many customers are from Brazil?
result_2 = customers.query("Country == 'Brazil'").shape[0]

#  From which three countries are the most customers?
result_3 = (
    customers.groupby("Country")
    .size()
    .sort_values(ascending=False)
    .head(3)
    .index.tolist()
)

# Which track has the longest length?
result_4 = tracks.sort_values("Milliseconds", ascending=False).iloc[0]["Name"]

# What is the average price of a track?
result_5 = tracks["UnitPrice"].mean()

#  How many genres are not represented by any tracks?
result_6 = genres[~genres["GenreId"].isin(tracks["GenreId"])].shape[0]

# How many distinct track composers are there?
result_7 = tracks["Composer"].nunique()

# Which artist has the most albums in the database?
result_8 = albums["ArtistId"].value_counts().index[0]

# How many customers does each employee support? show first and last name of employee
support_reps = customers[["SupportRepId"]]
merged_df = employees.merge(support_reps, left_on="EmployeeId", right_on="SupportRepId")
result_9 = merged_df.groupby(["FirstName", "LastName"]).size()


# Which artist has made the longest track?
album_id = tracks.sort_values("Milliseconds", ascending=False).iloc[0]["AlbumId"]
artist_id = albums[albums["AlbumId"] == album_id]["ArtistId"].iloc[0]
result_10 = artists[artists["ArtistId"] == artist_id]["Name"].iloc[0]

# How many tracks does the Grunge playlist have?
playlist_id = playlists[playlists["Name"] == "Grunge"]["PlaylistId"].iloc[0]
result_11 = playlist_track[playlist_track["PlaylistId"] == playlist_id].shape[0]

# What are the names of the three customers who spent the most money?
merged_df = customers.merge(invoices, on="CustomerId")
aggregated_df = merged_df.groupby("CustomerId")["Total"].sum()
customer_ids = aggregated_df.sort_values(ascending=False).head(3).index.tolist()
isin = customers["CustomerId"].isin(customer_ids)
result_12 = customers[isin][["FirstName", "LastName"]]

# What is name and purchase count of the media type that is the least popular amongst the tracks?
merged_df = tracks[["MediaTypeId"]].merge(media_types, on="MediaTypeId")
aggregated_df = merged_df.groupby("MediaTypeId")["Name"].count()
media_type_id = aggregated_df.sort_values().index[0]
result_13 = media_types[media_types["MediaTypeId"] == media_type_id]["Name"].iloc[0]

# Which artist has made the most revenue?
merged_df = tracks[["TrackId", "AlbumId"]].merge(invoice_items, on="TrackId")
merged_df = merged_df.merge(invoices, on="InvoiceId")
merged_df = merged_df.merge(albums, on="AlbumId")
merged_df = merged_df.merge(artists, on="ArtistId")
merged_df["TotalRevenue"] = merged_df["UnitPrice"] * merged_df["Quantity"]
result_14 = (
    merged_df.groupby("Name")["TotalRevenue"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

print("Done!")
