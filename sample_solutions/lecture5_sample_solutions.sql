-- How many albums are in the database? 
SELECT COUNT(*)
FROM albums;

-- How many customers are from Brazil? 
SELECT COUNT(*)
FROM customers
WHERE Country = 'Brazil';

-- From which three countries are the most customers? 
SELECT Country, COUNT(*) AS NumberOfCustomers
FROM customers
GROUP BY Country
ORDER BY NumberOfCustomers DESC
LIMIT 3;

-- Which track has the longest length? 
SELECT Name, Milliseconds
FROM tracks
ORDER BY Milliseconds DESC
LIMIT 1;

-- What is the average price of a track? 
SELECT AVG(UnitPrice) AS AveragePrice
FROM tracks;

-- How many genres are not represented by any tracks? 
SELECT COUNT(*) AS NumberOfGenres
FROM genres
WHERE GenreId NOT IN (SELECT GenreId
                      FROM tracks);

-- How many distinct track composers are there? 
SELECT COUNT(DISTINCT Composer) AS NumberOfComposers
FROM tracks;

-- Which artist has the most albums in the database? Solution using two queries: 
SELECT COUNT(*) AS NumberOfAlbums, ArtistId
FROM albums
GROUP BY ArtistId
ORDER BY NumberOfAlbums DESC
LIMIT 1;
SELECT Name
FROM artists
WHERE ArtistId = 90;

-- Which artist has the most albums in the database? Solution using one query: 
SELECT Name
FROM artists
WHERE ArtistId = (SELECT ArtistId
                  FROM albums
                  GROUP BY ArtistId
                  ORDER BY COUNT(*) DESC
                  LIMIT 1);

-- How many customers does each employee support? 
SELECT employees.FirstName, employees.LastName, COUNT(customers.SupportRepId) AS NumberOfCustomers
FROM employees
         LEFT JOIN customers ON employees.EmployeeId = customers.SupportRepId
GROUP BY employees.EmployeeId;

-- Which artist has made the longest track? Solution using three queries
SELECT MAX(Milliseconds) AS MaxMilliseconds, AlbumId
FROM tracks
GROUP BY AlbumId
ORDER BY MaxMilliseconds DESC
LIMIT 1;
SELECT ArtistId
from albums
WHERE AlbumId = 227;
SELECT Name
FROM artists
WHERE ArtistId = 147;

-- Which artist has made the longest track? Solution with one query: 
SELECT artists.Name, MAX(Milliseconds) AS MaxMilliseconds
FROM artists
         JOIN albums USING (ArtistId)
         JOIN tracks USING (AlbumId)
GROUP BY AlbumId
ORDER BY MAX(Milliseconds) DESC
LIMIT 1;

-- How many tracks does the Grunge playlist have? 
SELECT COUNT(*) AS NumberOfTracks
FROM playlist_track
WHERE PlaylistId = (SELECT PlaylistId
                    FROM playlists
                    WHERE Name = 'Grunge');

-- What are the names of the three customers who spent the most money?
SELECT customers.FirstName, customers.LastName, SUM(Total) AS Spendings
FROM customers
         JOIN invoices USING (CustomerId)
GROUP BY CustomerId
ORDER BY Spendings DESC
LIMIT 3;

-- Which media type is the least popular amongst the tracks? 
SELECT mt.Name AS MediaType, COUNT(*) AS NumberOfTracks
FROM tracks
         JOIN media_types mt USING (MediaTypeId)
GROUP BY MediaTypeId
ORDER BY NumberOfTracks
LIMIT 1;

-- Which artist has made the most revenue? 
SELECT artists.Name, SUM(invoice_items.UnitPrice * Quantity) AS Revenue
FROM artists
         JOIN albums USING (ArtistId)
         JOIN tracks USING (AlbumId)
         JOIN invoice_items USING (TrackId)
GROUP BY ArtistId
ORDER BY Revenue DESC
LIMIT 1;
